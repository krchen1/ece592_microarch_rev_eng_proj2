#define _GNU_SOURCE   /* turn on syscall(), CLOCK_MONOTONIC_RAW etc. before any header sees it */
#include <stdio.h>    /* printf for the CSV rows and the summary line */
#include <stdlib.h>   /* malloc/free, qsort, atoi, exit, posix_memalign, getenv, rand_r */
#include <stdint.h>   /* fixed-width ints: uint64_t counters, uintptr_t address math */
#include <string.h>   /* memset to fault every page in and to zero the perf attr struct */
#include <stdbool.h>  /* bool/true/false, kept available for the flag plumbing */

#define LINE_SIZE      64      /* bytes per cache line; set offsets stay multiples of this */
#define STRIDE         4096    /* one 4 KiB page between chain nodes so each lands in its own set */
#define WAYS_TO_TEST   8       /* default associativity to fill: one access per way */
#define ROUNDS         200000   /* ~1M+ probe pairs with the x15 vote (prof: "1M rep is a good start") */
#define CAL_ROUNDS     200000   /* per class; large sample => stable median/threshold */
#define CSV_SAMPLES    20        /* rows of the per-round trace dumped to the CSV */
#define VOTES          15       /* odd: majority vote per trial to beat per-probe noise */

#include <linux/perf_event.h>   /* perf_event_attr layout + the mmap page we read rdpmc from */
#include <sys/mman.h>           /* mmap() to map the perf counter page into our address space */
#include <sys/syscall.h>        /* __NR_perf_event_open: perf has no glibc wrapper */
#include <unistd.h>             /* syscall() and read() for the perf fd fallback path */
#include <time.h>               /* clock_gettime() for the portable non-x86/non-ARM timer */

static int g_fd = -1;                                  /* perf event fd; stays -1 unless perf is requested */
static struct perf_event_mmap_page *g_pc = NULL;       /* mmap'd perf page for userspace rdpmc; NULL if none */
static int g_have_rdpmc = 0;                           /* 1 once the kernel lets us read the counter in userspace */
static const char *TIMER_NAME = "(uninit)";            /* human label for whichever timer we settled on */

#if defined(__x86_64__) || defined(__i386__)   /* ---- Intel + AMD path: rdtscp / rdpmc ---- */
#include <x86intrin.h>   /* __rdtscp, _mm_mfence, _mm_lfence intrinsics */
static inline void serialize(void) { _mm_mfence(); _mm_lfence(); }   /* drain stores then loads before timing */
static inline uint64_t arch_pmc(uint32_t c) {   /* begin arch_pmc: read performance counter c via rdpmc */
    uint32_t a, d;   /* low 32 bits land in a (EAX), high 32 in d (EDX) */
    __asm__ volatile("lfence\n\trdpmc\n\tlfence" : "=a"(a), "=d"(d) : "c"(c));   /* fence, read counter #c, fence */
    return ((uint64_t)d << 32) | a;   /* stitch the two halves back into one 64-bit count */
}   /* end arch_pmc */
static inline uint64_t arch_fallback(void) {   /* begin arch_fallback: plain time-stamp counter */
    unsigned aux; _mm_lfence(); uint64_t t = __rdtscp(&aux); _mm_lfence(); return t;   /* fenced rdtscp read */
}   /* end arch_fallback */
static const char *FALLBACK_NAME = "rdtscp";   /* label used when we take the plain-TSC path */
#elif defined(__aarch64__)   /* ---- ARM Neoverse N1 path: PMU / virtual counter ---- */
static inline void serialize(void) { __asm__ volatile("dsb sy; isb" ::: "memory"); }   /* full barrier + pipeline flush */
static inline uint64_t arch_pmc(uint32_t c) {   /* begin arch_pmc: read an ARM PMU counter */
    uint64_t v;   /* holds the counter value we read out */
    __asm__ volatile("isb" ::: "memory");   /* serialize so we sample at a defined point */
    if (c == 31) {                      /* ARMv8 cycle counter */
        __asm__ volatile("mrs %0, pmccntr_el0" : "=r"(v));   /* read the dedicated cycle register */
    } else {                            /* programmable event counter c */
        __asm__ volatile("msr pmselr_el0, %1\n\tisb\n\tmrs %0, pmxevcntr_el0"   /* select counter c, then read it */
                         : "=r"(v) : "r"((uint64_t)c));   /* out: v; in: the counter index */
    }   /* end counter-select if */
    return v;   /* hand back whichever counter we read */
}   /* end arch_pmc */
static inline uint64_t arch_fallback(void) {   /* begin arch_fallback: EL0-readable virtual counter */
    uint64_t v; __asm__ volatile("isb; mrs %0, cntvct_el0; isb" : "=r"(v)); return v;   /* fenced cntvct read */
}   /* end arch_fallback */
static const char *FALLBACK_NAME = "cntvct_el0";   /* label used when we take the virtual-counter path */
#else   /* ---- anything else: portable clock, no cycle counter ---- */
static inline void serialize(void) { __sync_synchronize(); }   /* generic full memory barrier */
static inline uint64_t arch_pmc(uint32_t c) { (void)c; return 0; }   /* no PMU here; swallow arg, return nothing */
static inline uint64_t arch_fallback(void) {   /* begin arch_fallback: monotonic clock in nanoseconds */
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);   /* raw clock, unaffected by NTP slew */
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;   /* collapse to a single ns count */
}   /* end arch_fallback */
static const char *FALLBACK_NAME = "CLOCK_MONOTONIC_RAW";   /* label used on the portable path */
#endif   /* architecture select */

static void timer_init(void) {   /* begin timer_init: decide which timer rdcycle() will use */
    /* The self-referenced ratio (atk/ref) already cancels DVFS, so the plain
       arch counter (rdtsc/cntvct) is enough and is markedly more stable here
       than the perf rdpmc read. Use perf only if explicitly requested. */
    if (!getenv("PRIMEPROBE_PERF")) { TIMER_NAME = FALLBACK_NAME; return; }   /* default: skip perf, use the arch counter */
    struct perf_event_attr pe;   /* describes the counter we want the kernel to open */
    memset(&pe, 0, sizeof(pe));   /* zero it so every field we don't set reads as 0 */
    pe.type = PERF_TYPE_HARDWARE;   /* a generic hardware event... */
    pe.size = sizeof(pe);   /* tell the kernel which struct version we're passing */
    pe.config = PERF_COUNT_HW_CPU_CYCLES;   /* ...specifically the CPU cycle count */
    pe.exclude_kernel = 1;   /* don't count cycles spent in the kernel */
    pe.exclude_hv = 1;   /* don't count hypervisor cycles either */
    g_fd = (int)syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);   /* open it for this thread on any CPU */
    if (g_fd < 0) { TIMER_NAME = FALLBACK_NAME; return; }   /* kernel said no -> fall back to the arch counter */
    g_pc = (struct perf_event_mmap_page *)mmap(NULL, 4096, PROT_READ, MAP_SHARED, g_fd, 0);   /* map the counter page */
    if (g_pc == MAP_FAILED) { g_pc = NULL; TIMER_NAME = "perf cpu-cycles via read()"; return; }   /* no mmap -> read() path */
    g_have_rdpmc = g_pc->cap_user_rdpmc ? 1 : 0;   /* can we read the counter straight from userspace? */
    TIMER_NAME = g_have_rdpmc ? "perf cpu-cycles (rdpmc)" : "perf cpu-cycles via read()";   /* label the chosen path */
}   /* end timer_init */

static inline uint64_t rdcycle(void) {   /* begin rdcycle: return the current cycle/time count */
    if (g_have_rdpmc) {   /* fast path: read the perf counter from userspace */
        uint32_t seq, idx, width; uint64_t count;   /* seqlock token, counter index, bit width, running offset */
        do {   /* retry until we read a consistent snapshot (seqlock) */
            seq = g_pc->lock;   /* grab the sequence number before reading */
            __asm__ volatile("" ::: "memory");   /* compiler barrier: don't hoist the reads above the lock grab */
            idx = g_pc->index; count = g_pc->offset; width = g_pc->pmc_width;   /* kernel's view of the counter */
            if (idx) {   /* idx==0 means the counter isn't currently active; skip the rdpmc then */
                uint64_t v = arch_pmc(idx - 1);   /* read raw hardware counter (index is 1-based here) */
                v <<= (64 - width); count += (uint64_t)((int64_t)v >> (64 - width));   /* sign-extend to full width, add */
            }   /* end active-counter if */
            __asm__ volatile("" ::: "memory");   /* compiler barrier before re-checking the lock */
        } while (g_pc->lock != seq);   /* lock changed under us -> the snapshot was torn, read again */
        return count;   /* consistent counter value */
    }   /* end rdpmc fast path */
    if (g_fd >= 0) { uint64_t v; if (read(g_fd, &v, sizeof(v)) == (ssize_t)sizeof(v)) return v; }   /* slow perf read() */
    return arch_fallback();   /* no perf at all -> plain arch counter / clock */
}   /* end rdcycle */

/* set_offset shifts the chain into a different cache set (keep it a multiple of
   LINE_SIZE and smaller than stride). 0 = same set as another chain built with 0. */
static void **build_chain(size_t ways, size_t stride, size_t set_offset, void **raw_out) {   /* begin build_chain */
    size_t total = ways * stride + 2 * stride;   /* room for every node plus slack for alignment + offset */
    void *buf = NULL;   /* the raw allocation we'll carve the chain out of */
    if (posix_memalign(&buf, stride, total) != 0) { perror("posix_memalign"); exit(1); }   /* stride-aligned or die */
    memset(buf, 1, total);   /* touch every page so it's really faulted in, not lazily zero-mapped */
    *raw_out = buf;   /* hand the raw pointer back so the caller can free() it later */

    uintptr_t base = (((uintptr_t)buf + stride - 1) & ~((uintptr_t)stride - 1)) + set_offset;   /* align up, add offset */
    for (size_t w = 0; w < ways; w++) {   /* lay out one node per way */
        void **node = (void **)(base + w * stride);   /* this way's node sits one full stride past the last */
        *node = (void *)(base + ((w + 1) % ways) * stride);   /* point it at the next way, wrapping into a ring */
    }   /* end way loop */
    return (void **)base;   /* the head of the circular pointer chain */
}   /* end build_chain */

static inline void walk(void **head, size_t steps) {   /* begin walk: chase the chain to prime/evict a set */
    void **p = head;   /* start at the head node */
    for (size_t i = 0; i < steps; i++) p = (void **)(*p);   /* follow each node to the next for `steps` hops */
    __asm__ volatile("" :: "r"(p) : "memory");   /* keep p live so the compiler can't delete the whole walk */
}   /* end walk */

static int cmp_u64(const void *a, const void *b) {   /* begin cmp_u64: qsort comparator for uint64_t */
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;   /* reinterpret the two void* as uint64_t */
    return (x > y) - (x < y);   /* branch-free -1/0/+1 without risking overflow of a subtraction */
}   /* end cmp_u64 */

/* # of sorted-ascending elements strictly greater than t */
static size_t count_gt(const uint64_t *a, size_t n, uint64_t t) {   /* begin count_gt: FP count if cut at t */
    size_t lo = 0, hi = n;   /* binary-search bounds over the sorted array */
    while (lo < hi) { size_t m = (lo + hi) / 2; if (a[m] > t) hi = m; else lo = m + 1; }   /* find first element > t */
    return n - lo;   /* everything from lo..n-1 is strictly greater than t */
}   /* end count_gt */
/* # of sorted-ascending elements <= t */
static size_t count_le(const uint64_t *a, size_t n, uint64_t t) {   /* begin count_le: FN count if cut at t */
    size_t lo = 0, hi = n;   /* binary-search bounds over the sorted array */
    while (lo < hi) { size_t m = (lo + hi) / 2; if (a[m] <= t) lo = m + 1; else hi = m; }   /* find first element > t */
    return lo;   /* everything before lo is <= t */
}   /* end count_le */

/* Choose the threshold (ratio in permille) that minimizes calibration error
   FP(hit>t)+FN(miss<=t). The mean-of-medians rule misplaces the cut when the
   two clusters are asymmetric or overlapping (e.g. AMD Zen 2's coarsely
   quantized small-cycle walks); picking the empirically best cut handles that.
   For a cleanly separated channel the whole gap is error-free, so this returns
   the centre of that gap -- identical to the old rule. hit/miss are sorted. */
static uint64_t best_threshold(const uint64_t *hit, const uint64_t *miss, size_t n) {   /* begin best_threshold */
    uint64_t lo = hit[0], hi = miss[n - 1];   /* scan from the smallest hit up to the largest miss */
    if (hi <= lo) return lo + 1;   /* classes don't overlap at all -> any cut just below works */
    size_t best_err = (size_t)-1;   /* best (lowest) error seen so far; start at +infinity */
    uint64_t first_best = lo, last_best = lo;   /* track the span of thresholds sharing that best error */
    for (uint64_t t = lo; t <= hi; t++) {   /* try every candidate cut across the overlap */
        size_t err = count_gt(hit, n, t) + count_le(miss, n, t);   /* misclassified hits + misclassified misses */
        if (err < best_err) { best_err = err; first_best = last_best = t; }   /* new best -> reset the plateau */
        else if (err == best_err) { last_best = t; }   /* ties extend the current best-error plateau */
    }   /* end candidate loop */
    return (first_best + last_best) / 2;     /* centre of the best-error plateau */
}   /* end best_threshold */

/* Time the whole dependent walk once. Pointer-chasing serializes the accesses
   by data dependency, so a single fence + two counter reads amortize the timer
   overhead across all `ways` accesses, keeping the hit/miss gap resolvable. */
static uint64_t probe_time(void **head, size_t ways) {   /* begin probe_time: cycles for one full chain walk */
    void **p = head;   /* start at the head node */
    serialize();   /* make sure nothing earlier is still in flight before we start the clock */
    uint64_t t0 = rdcycle();   /* open the timing window */
    for (size_t w = 0; w < ways; w++) p = (void **)(*p);   /* chase all `ways` dependent loads back to back */
    uint64_t t1 = rdcycle();   /* close the timing window */
    __asm__ volatile("" :: "r"(p) : "memory");   /* keep p live so the walk isn't optimized away */
    return t1 - t0;   /* elapsed cycles for the whole walk */
}   /* end probe_time */

int main(int argc, char **argv) {   /* begin main */
    size_t ways = WAYS_TO_TEST;   /* associativity to fill; overridable from the command line */
    size_t stride = STRIDE;   /* byte gap between nodes; overridable from the command line */
    if (argc > 1) ways = (size_t)atoi(argv[1]);   /* argv[1] overrides ways (match this CPU's L1d) */
    if (argc > 2) stride = (size_t)atoi(argv[2]);   /* argv[2] overrides stride */

    void *raw_atk, *raw_vic_conf, *raw_vic_far, *raw_ref;   /* raw malloc bases, kept only so we can free them */
    timer_init();   /* pick rdtscp/cntvct/perf and set TIMER_NAME */

    void **atk = build_chain(ways, stride, 0, &raw_atk);                     /* attacker's eviction set (set 0) */
    void **vic_conf = build_chain(ways, stride, 0, &raw_vic_conf);           /* same set as attacker */
    void **ref      = build_chain(ways, stride, stride / 4, &raw_ref);       /* reference set, never evicted */
    void **vic_far  = build_chain(ways, stride, stride / 2, &raw_vic_far);   /* different set */

    /* Warm up the core so DVFS has ramped to a steady clock before calibrating. */
    for (int r = 0; r < 200000; r++) walk(atk, ways);   /* burn walks to pull the clock up to speed */

    /* Classify by the ratio probe(attacker)/probe(reference), measured back to
       back each round. The constant-rate TSC inflates both walks equally when
       the core down-clocks, so the ratio cancels DVFS drift that a fixed
       absolute threshold cannot. Interleave hit/miss sampling for calibration. */
    uint64_t *hit_samp = malloc((size_t)CAL_ROUNDS * sizeof(uint64_t));   /* calibration samples for the "hit" class */
    uint64_t *miss_samp = malloc((size_t)CAL_ROUNDS * sizeof(uint64_t));  /* calibration samples for the "miss" class */
    if (!hit_samp || !miss_samp) { perror("malloc"); exit(1); }   /* no buffers -> nothing to calibrate */
    for (int r = 0; r < CAL_ROUNDS; r++) {   /* collect one hit and one miss sample per round */
        walk(atk, ways); walk(ref, ways);   /* prime: fill both the attacker set and the reference set */
        uint64_t tr = probe_time(ref, ways), ta = probe_time(atk, ways);   /* time the reference then the attacker */
        hit_samp[r] = tr ? (ta * 1000) / tr : 1000;                 /* hit: ratio ~1000 */
        walk(atk, ways); walk(ref, ways); walk(vic_conf, ways);   /* re-prime, then let the victim evict a way */
        tr = probe_time(ref, ways); ta = probe_time(atk, ways);   /* time both again after the eviction */
        miss_samp[r] = tr ? (ta * 1000) / tr : 1000;                /* miss: ratio > 1000 */
    }   /* end calibration loop */
    qsort(hit_samp, CAL_ROUNDS, sizeof(uint64_t), cmp_u64);   /* sort hits so we can take medians/percentiles */
    qsort(miss_samp, CAL_ROUNDS, sizeof(uint64_t), cmp_u64);   /* sort misses the same way */
    uint64_t hit_med = hit_samp[CAL_ROUNDS / 2];   /* median hit ratio */
    uint64_t miss_med = miss_samp[CAL_ROUNDS / 2];   /* median miss ratio */
    uint64_t threshold = best_threshold(hit_samp, miss_samp, CAL_ROUNDS);   /* cut that minimizes calibration error */
    if (threshold <= hit_samp[0]) threshold = hit_samp[0] + 1;   /* never cut below the smallest hit */


    if (getenv("PRIMEPROBE_DUMP")) {   /* debug mode: dump raw samples for the histogram instead of running */
        for (int r = 0; r < CAL_ROUNDS; r++) printf("hit %llu\n", (unsigned long long)hit_samp[r]);   /* all hits */
        for (int r = 0; r < CAL_ROUNDS; r++) printf("miss %llu\n", (unsigned long long)miss_samp[r]);  /* all misses */
        free(hit_samp); free(miss_samp);   /* release the sample buffers */
        return 0;   /* dump done, nothing else to do */
    }   /* end dump mode */

    printf("# Prime+Probe leakage trace (controlled victim, known secret)\n");   /* CSV banner line */
    printf("# ways=%zu stride=%zu timer=%s\n", ways, stride, TIMER_NAME);   /* record the run's geometry + timer */
    printf("# hit_ratio  p5/p50/p95 = %llu.%03llu / %llu.%03llu / %llu.%03llu (atk/ref)\n",   /* hit spread */
           (unsigned long long)hit_samp[CAL_ROUNDS/20]/1000, (unsigned long long)hit_samp[CAL_ROUNDS/20]%1000,   /* p5 */
           (unsigned long long)hit_med/1000, (unsigned long long)hit_med%1000,   /* p50 (median) */
           (unsigned long long)hit_samp[CAL_ROUNDS*19/20]/1000, (unsigned long long)hit_samp[CAL_ROUNDS*19/20]%1000);  /* p95 */
    printf("# miss_ratio p5/p50/p95 = %llu.%03llu / %llu.%03llu / %llu.%03llu (atk/ref)\n",   /* miss spread */
           (unsigned long long)miss_samp[CAL_ROUNDS/20]/1000, (unsigned long long)miss_samp[CAL_ROUNDS/20]%1000,   /* p5 */
           (unsigned long long)miss_med/1000, (unsigned long long)miss_med%1000,   /* p50 (median) */
           (unsigned long long)miss_samp[CAL_ROUNDS*19/20]/1000, (unsigned long long)miss_samp[CAL_ROUNDS*19/20]%1000);  /* p95 */
    printf("# threshold=%llu.%03llu (atk/ref)\n",   /* the frozen decision threshold */
           (unsigned long long)threshold / 1000, (unsigned long long)threshold % 1000);   /* split permille into int.frac */
    printf("round,secret,ratio_milli,detected\n");   /* CSV column header for the per-round trace */

    unsigned seed = 12345;   /* fixed seed so the secret stream is reproducible across runs */
    uint64_t tp = 0, fp = 0, tn = 0, fn = 0;   /* confusion-matrix counters */
    int sample_every = ROUNDS / CSV_SAMPLES; if (sample_every < 1) sample_every = 1;   /* dump every Nth round */
    for (int r = 0; r < ROUNDS; r++) {   /* the actual attack: one secret bit per round */
        int secret = rand_r(&seed) & 1;   /* ground-truth bit: 1 = victim touches our set, 0 = far set */
        int yes = 0;   /* how many of the VOTES probes said "evicted" */
        uint64_t last_ratio = 0;   /* ratio from the final vote, kept for the CSV row */
        for (int v = 0; v < VOTES; v++) {   /* majority-vote several probes to beat per-probe noise */
            walk(atk, ways); walk(ref, ways);   /* prime the attacker set and the reference set */
            walk(secret ? vic_conf : vic_far, ways);   /* victim runs: conflicting set if secret, far set if not */
            uint64_t tr = probe_time(ref, ways), ta = probe_time(atk, ways);   /* time reference then attacker */
            last_ratio = tr ? (ta * 1000) / tr : 1000;   /* attacker/reference ratio in permille */
            if (last_ratio > threshold) yes++;   /* over the cut -> this vote calls it an eviction */
        }   /* end vote loop */
        int detected = (yes * 2 > VOTES) ? 1 : 0;   /* majority of votes -> we declare the set was touched */

        if (secret && detected) tp++;   /* touched and we caught it */
        else if (secret && !detected) fn++;   /* touched but we missed it */
        else if (!secret && detected) fp++;   /* not touched but we cried wolf */
        else tn++;   /* not touched and we correctly stayed quiet */

        if (r % sample_every == 0)   /* only dump a sampled subset of rounds to keep the CSV small */
            printf("%d,%d,%llu,%d\n", r, secret, (unsigned long long)last_ratio, detected);   /* one trace row */
    }   /* end attack loop */

    double acc = 100.0 * (double)(tp + tn) / ROUNDS;   /* overall accuracy = correct / total */
    double fpr = (fp + tn) ? 100.0 * (double)fp / (fp + tn) : 0.0;   /* false-positive rate over the secret-0 trials */
    double fnr = (tp + fn) ? 100.0 * (double)fn / (tp + fn) : 0.0;   /* false-negative rate over the secret-1 trials */
    printf("\n# summary: accuracy=%.2f%% fp_rate=%.2f%% fn_rate=%.2f%% "   /* final scoreboard line... */
           "tp=%llu tn=%llu fp=%llu fn=%llu\n",   /* ...plus the raw confusion counts */
           acc, fpr, fnr,   /* the three rates */
           (unsigned long long)tp, (unsigned long long)tn,   /* correct cells */
           (unsigned long long)fp, (unsigned long long)fn);   /* error cells */

    free(hit_samp); free(miss_samp);   /* release the calibration buffers */
    free(raw_atk); free(raw_vic_conf); free(raw_vic_far); free(raw_ref);   /* release the four chain allocations */
    return 0;   /* clean exit */
}   /* end main */
