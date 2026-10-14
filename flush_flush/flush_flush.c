/* flush_flush_balanced.c */

#include <stdint.h>   // fixed-width integer types (uint64_t etc.)
#include <stdio.h>    // printf / fprintf for CSV + summaries
#include <stdlib.h>   // posix_memalign, malloc, free, rand, srand, strtol, exit
#include <string.h>   // memset to fault pages in, strcmp for mode dispatch
#include <time.h>     // time() to seed rand(), clock_gettime() for wall time
#include <unistd.h>   // getpid() to decorrelate the per-run seeds
#include <math.h>     // log2() for the information-theoretic leakage rate (link with -lm)
#include <sys/resource.h>   // getrusage() for peak resident memory (ru_maxrss)

/* ----- tunable constants ---------------------------------------------- */
#define CACHE_LINE   64      /* bytes per cache line; probes must sit on separate lines */
#define STRIDE       4096    /* one 4 KiB page between probe lines so they never share a line/prefetch stream */
#define NUM_SAMPLES  10000   /* samples collected PER CLASS during calibration */

/* ====================================================================== */
/* Architecture-abstracted primitives: counter read, flush, and fences.   */
/* Each target gets the same C-level API so the experiment code below is   */
/* written once.                                                          */
/* ====================================================================== */
#if defined(__x86_64__)   /* ---- Intel + AMD path: TSC + clflush ------- */

/* tsc_start(): read the Time-Stamp Counter to OPEN a timing window. */
static inline uint64_t tsc_start(void)
{   /* begin tsc_start */
    uint32_t lo;   /* low 32 bits of the 64-bit counter */
    uint32_t hi;   /* high 32 bits of the 64-bit counter */
    asm volatile (
        "lfence\n\t"   /* drain earlier loads so pre-window work can't leak into the measurement */
        "rdtsc\n\t"    /* read TSC: low 32 -> EAX, high 32 -> EDX */
        : "=a"(lo),    /* output: EAX into lo */
          "=d"(hi)     /* output: EDX into hi */
        :              /* no inputs */
        : "memory"     /* clobber: forbid reordering memory ops across this */
    );
    return ((uint64_t)hi << 32) | lo;   /* widen hi first, then shift+OR so no bits fall off */
}   /* end tsc_start */

/* tsc_end(): read the TSC to CLOSE the window. */
static inline uint64_t tsc_end(void)
{   /* begin tsc_end */
    uint32_t lo;   /* low 32 bits */
    uint32_t hi;   /* high 32 bits */
    asm volatile (
        "rdtscp\n\t"   /* read TSC only after all earlier instructions have retired */
        "lfence\n\t"   /* stop later instructions from sliding back before the read */
        : "=a"(lo),    /* EAX into lo */
          "=d"(hi)     /* EDX into hi */
        :              /* no inputs */
        : "rcx", "memory"   /* rdtscp also writes ECX, so declare rcx clobbered */
    );
    return ((uint64_t)hi << 32) | lo;   /* reassemble the 64-bit value */
}   /* end tsc_end */

/* flush(addr): evict the one cache line containing addr from all levels. */
static inline void flush(volatile void *addr)
{   /* begin flush */
    asm volatile (
        "clflush (%0)\n\t"   /* flush the line at the address in operand 0 */
        :                    /* no outputs */
        : "r"(addr)          /* input: addr in any general register */
        : "memory"           /* memory state changes invisibly to the compiler */
    );
}   /* end flush */

/* fence_full(): full barrier so the flush actually lands before we time it. */
static inline void fence_full(void)
{   /* begin fence_full */
    asm volatile ("mfence\n\t" : : : "memory");   /* serialize all loads+stores */
}   /* end fence_full */

#define ARCH_NAME "x86_64"   /* label emitted in the CSV header */

#elif defined(__aarch64__)   /* ---- ARM Neoverse N1 path ---------------- */

/* tsc_start(): read a monotonic counter to OPEN the timing window. */
static inline uint64_t tsc_start(void)
{   /* begin tsc_start */
    uint64_t v;   /* holds the 64-bit counter value */
#ifdef USE_PMCCNTR   /* optional cycle-accurate path, needs userspace PMU access enabled */
    asm volatile (
        "isb\n\t"                 /* flush the pipeline so the read samples at a defined point */
        "mrs %0, pmccntr_el0\n\t"  /* read the cycle counter (fine resolution) */
        "isb\n\t"                 /* block later instructions from being sampled early */
        : "=r"(v) : : "memory"
    );
#else                /* default portable path: virtual count register, EL0-readable on Linux */
    asm volatile (
        "isb\n\t"                 /* serialize before sampling the counter */
        "mrs %0, cntvct_el0\n\t"   /* read the virtual counter (COARSE: ~tens of MHz tick) */
        "isb\n\t"                 /* serialize after */
        : "=r"(v) : : "memory"
    );
#endif
    return v;   /* hand back the counter value */
}   /* end tsc_start */

/* tsc_end(): read the same counter to CLOSE the window. */
static inline uint64_t tsc_end(void)
{   /* begin tsc_end */
    uint64_t v;   /* counter value */
#ifdef USE_PMCCNTR   /* match whichever counter tsc_start used */
    asm volatile (
        "isb\n\t"                 /* make sure the flush has retired before we read */
        "mrs %0, pmccntr_el0\n\t"  /* read the cycle counter */
        "isb\n\t"                 /* serialize after */
        : "=r"(v) : : "memory"
    );
#else
    asm volatile (
        "isb\n\t"                 /* serialize before */
        "mrs %0, cntvct_el0\n\t"   /* read the virtual counter */
        "isb\n\t"                 /* serialize after */
        : "=r"(v) : : "memory"
    );
#endif
    return v;   /* hand back the counter value */
}   /* end tsc_end */

/* flush(addr): clean+invalidate the line containing addr to the point of coherency. */
static inline void flush(volatile void *addr)
{   /* begin flush */
    asm volatile (
        "dc civac, %0\n\t"   /* Data Cache Clean & Invalidate by VA to PoC (EL0 needs SCTLR_EL1.UCI, set by Linux) */
        :                    /* no outputs */
        : "r"(addr)          /* input: the virtual address */
        : "memory"           /* cache state changes invisibly to the compiler */
    );
}   /* end flush */

/* fence_full(): full data barrier so DC CIVAC completes before timing. */
static inline void fence_full(void)
{   /* begin fence_full */
    asm volatile ("dsb ish\n\t" "isb\n\t" : : : "memory");   /* drain to inner-shareable, then flush pipeline */
}   /* end fence_full */

#define ARCH_NAME "aarch64"   /* label emitted in the CSV header */

#else   /* ---- any other architecture is unsupported ------------------- */
#error "unsupported architecture: need x86_64 (clflush/rdtsc) or aarch64 (dc civac/cntvct)"
#endif  /* architecture select */

/* time_flush(addr): fenced measurement of how long it takes to flush addr. */
static uint64_t time_flush(volatile uint8_t *addr)
{   /* begin time_flush */
    uint64_t t0, t1;         /* start/end counter readings */
    fence_full();            /* make sure nothing earlier is still in flight */
    t0 = tsc_start();        /* open the timing window */
    flush((void *)addr);     /* the operation we are timing */
    fence_full();            /* wait for the flush to actually land */
    t1 = tsc_end();          /* close the timing window */
    return t1 - t0;          /* elapsed counter ticks for this flush */
}   /* end time_flush */

/* ====================================================================== */
/* Shared helpers: table layout, victim touch, median, shuffle.           */
/* ====================================================================== */

/* Two class positions in the table: line 0 is the one we (maybe) touch,   */
/* line 1 is a never-touched control used to balance the layout.           */
#define NUM_LINES 2   /* how many distinct probe lines the table holds */

/* victim(table, line): perform the single "secret" memory access. */
static void victim(volatile uint8_t *table, int line)
{   /* begin victim */
    (void)table[line * STRIDE];   /* one touch of the chosen line pulls it into cache */
}   /* end victim */

/* median_of(): insertion-sort in place and return the middle element. */
static uint64_t median_of(uint64_t *values, int n)
{   /* begin median_of */
    for (int i = 1; i < n; i++) {           /* i walks the unsorted tail */
        uint64_t key = values[i];            /* value being inserted into the sorted head */
        int j = i - 1;                        /* scan leftward through the sorted head */
        while (j >= 0 && values[j] > key) {  /* shift anything larger than key one slot right */
            values[j + 1] = values[j];          /* slide it over */
            j--;                                  /* keep walking left */
        }   /* end while */
        values[j + 1] = key;                 /* drop key into its sorted slot */
    }   /* end for */
    return values[n / 2];   /* middle-ish element once sorted (good enough for a threshold) */
}   /* end median_of */

/* shuffle_order(): Fisher-Yates shuffle of order[0..n-1]. */
static void shuffle_order(int *order, int n)
{   /* begin shuffle_order */
    for (int i = n - 1; i > 0; i--) {   /* walk from the end, swapping each slot with an earlier random one */
        int j = rand() % (i + 1);        /* pick the swap partner in [0, i] */
        int tmp = order[i];               /* standard three-way swap */
        order[i] = order[j];              /* first half of the swap */
        order[j] = tmp;                    /* second half of the swap */
    }   /* end for */
}   /* end shuffle_order */

/* alloc_table(): allocate+map the page-aligned probe table, or die. */
static uint8_t *alloc_table(void)
{   /* begin alloc_table */
    uint8_t *table;   /* pointer to the allocated buffer */
    if (posix_memalign((void **)&table, 4096, (size_t)NUM_LINES * STRIDE) != 0) {   /* page-aligned alloc */
        perror("posix_memalign");   /* report why it failed */
        exit(1);                     /* no table, nothing to do */
    }   /* end if */
    memset(table, 1, (size_t)NUM_LINES * STRIDE);   /* write every page so it is actually faulted in */
    return table;   /* hand back the ready-to-use table */
}   /* end alloc_table */

/* class labels used by both calibration and measurement */
#define LABEL_MISS 0   /* line was NOT touched before we flush+time it */
#define LABEL_HIT  1   /* line WAS touched (cached) before we flush+time it */

/* warmup iterations burned before recording, to get past clock ramp /    */
/* cache+TLB warmup that otherwise drags the early samples slower.         */
#define CALIB_WARMUP_ITERS 3000   /* throwaway iterations before calibration records anything */
#define MEASURE_WARMUP_ITERS 3000   /* throwaway iterations before the measure runs, same purpose as calib warmup */

/* ====================================================================== */
/* calibrate -> 10000 hit + 10000 miss samples, suggest threshold */
/* ====================================================================== */
static void calibrate(void)
{   /* begin calibrate */
    srand((unsigned)time(NULL));   /* seed the RNG once for this calibration run */

    uint8_t *table = alloc_table();   /* same table shape the measure path uses, so the threshold transfers */

    uint64_t *hit_samples  = malloc(NUM_SAMPLES * sizeof(uint64_t));   /* buffer for the hit class */
    uint64_t *miss_samples = malloc(NUM_SAMPLES * sizeof(uint64_t));   /* buffer for the miss class */
    if (!hit_samples || !miss_samples) {   /* either allocation failing means no calibration */
        perror("malloc");   /* report the failure */
        exit(1);             /* cannot continue without the buffers */
    }   /* end if */

    for (int w = 0; w < CALIB_WARMUP_ITERS; w++) {   /* warmup: identical work, but discard the timings */
        flush((volatile uint8_t *)&table[0 * STRIDE]);   /* evict the probe line */
        fence_full();                                     /* make sure the flush landed */
        victim(table, 0);                                 /* touch it so the warmup mirrors a hit trial */
        (void)time_flush(&table[0 * STRIDE]);             /* time and throw the result away */
    }   /* end warmup loop */

    for (int i = 0; i < NUM_SAMPLES; i++) {   /* collect one HIT sample per iteration */
        flush((volatile uint8_t *)&table[0 * STRIDE]);   /* start from an evicted (uncached) line */
        fence_full();                                     /* ensure the eviction completed */
        victim(table, 0);                                 /* touch the line so it is now cached */
        uint64_t t = time_flush(&table[0 * STRIDE]);      /* time the flush of the cached line */
        hit_samples[i] = t;                               /* record the hit-class sample */
        printf("hit %lu\n", (unsigned long)t);            /* emit it (compatible with the analyzer) */
    }   /* end hit loop */

    for (int i = 0; i < NUM_SAMPLES; i++) {   /* collect one MISS sample per iteration */
        flush((volatile uint8_t *)&table[0 * STRIDE]);   /* evict the line */
        fence_full();                                     /* ensure the eviction completed */
        /* deliberately do NOT touch the line, so it stays uncached */
        uint64_t t = time_flush(&table[0 * STRIDE]);      /* time the flush of the uncached line */
        miss_samples[i] = t;                              /* record the miss-class sample */
        printf("miss %lu\n", (unsigned long)t);           /* emit it (compatible with the analyzer) */
    }   /* end miss loop */

    free(table);   /* done with the probe table */

    uint64_t hit_median  = median_of(hit_samples, NUM_SAMPLES);    /* typical hit timing (sorts in place) */
    uint64_t miss_median = median_of(miss_samples, NUM_SAMPLES);   /* typical miss timing (sorts in place) */
    int threshold = (int)((hit_median + miss_median) / 2);          /* freeze the midpoint between the classes */

    /* Polarity is derived, not assumed: on some parts a cached line flushes
     * SLOWER (hit above threshold), on others FASTER (hit below). Pick the
     * side the hit median actually sits on. */
    const char *polarity = (hit_median > miss_median) ? "above" : "below";   /* "above" => hit when timing > T */

    uint32_t misclassified = 0;   /* count samples the frozen threshold would get wrong */
    for (int i = 0; i < NUM_SAMPLES; i++) {   /* check every hit sample */
        int pred_hit = (hit_median > miss_median) ? ((int)hit_samples[i] > threshold)   /* classify with the chosen polarity */
                                                   : ((int)hit_samples[i] < threshold);
        if (!pred_hit) misclassified++;   /* a hit sample classified as miss is an error */
    }   /* end hit check */
    for (int i = 0; i < NUM_SAMPLES; i++) {   /* check every miss sample */
        int pred_hit = (hit_median > miss_median) ? ((int)miss_samples[i] > threshold)   /* same polarity rule */
                                                   : ((int)miss_samples[i] < threshold);
        if (pred_hit) misclassified++;   /* a miss sample classified as hit is an error */
    }   /* end miss check */

    fprintf(stderr,   /* frozen-threshold summary goes to stderr so stdout stays pure samples */
        "FROZEN threshold=%d polarity=%s hit_median=%lu miss_median=%lu "
        "misclassified=%u/%d (%.2f%%)\n",
        threshold, polarity, (unsigned long)hit_median, (unsigned long)miss_median,
        misclassified, 2 * NUM_SAMPLES, 100.0 * misclassified / (2 * NUM_SAMPLES));   /* the numbers */

    free(hit_samples);    /* cleanup */
    free(miss_samples);   /* cleanup */
}   /* end calibrate */

/* mi_bits(): mutual information I(label;prediction) in bits/trial, derived  */
/* from the 2x2 confusion matrix. This is the true LEAKAGE per trial of a    */
/* noisy binary channel: 1.0 bit means a perfect oracle, 0.0 means the       */
/* prediction tells you nothing about the secret. Zero-count cells contribute */
/* nothing (0*log0 == 0 by convention). */
static double mi_bits(uint64_t tp, uint64_t tn, uint64_t fp, uint64_t fn)
{   /* begin mi_bits */
    double n = (double)(tp + tn + fp + fn);   /* total trials */
    if (n == 0.0) return 0.0;                   /* no data => no information */

    double px_hit  = (double)(tp + fn) / n;   /* P(label=hit) */
    double px_miss = (double)(fp + tn) / n;   /* P(label=miss) */
    double py_hit  = (double)(tp + fp) / n;   /* P(pred=hit) */
    double py_miss = (double)(fn + tn) / n;   /* P(pred=miss) */

    double cells[4] = { (double)tp, (double)fn, (double)fp, (double)tn };   /* joint counts */
    double px[4]    = { px_hit, px_hit, px_miss, px_miss };                  /* matching P(label) per cell */
    double py[4]    = { py_hit, py_miss, py_hit, py_miss };                  /* matching P(pred)  per cell */

    double mi = 0.0;   /* accumulated mutual information */
    for (int c = 0; c < 4; c++) {   /* sum over the four confusion cells */
        if (cells[c] == 0.0) continue;         /* skip empty cells (0*log0 == 0) */
        double pxy = cells[c] / n;              /* joint probability P(label,pred) */
        double denom = px[c] * py[c];           /* independence baseline P(label)P(pred) */
        if (denom > 0.0)                        /* guard against a zero marginal */
            mi += pxy * log2(pxy / denom);      /* this cell's contribution to I(X;Y) */
    }   /* end for */
    return mi;   /* bits of information leaked per trial */
}   /* end mi_bits */

/* ====================================================================== */
/* measure -> 10 runs x 10000 fresh balanced trials, frozen T.    */
/* Collects, per run: leakage rate, accuracy, false pos/neg, latency,     */
/* one-time setup cost, and memory use.                                   */
/* ====================================================================== */
static void measure(int threshold, int hit_above)
{   /* begin measure (hit_above: 1 => predict hit when timing > T, else <) */
    const int RUNS           = 10;      /* independent runs */
    const int TRIALS_PER_RUN = 10000;   /* trials per run */
    const int HALF           = TRIALS_PER_RUN / 2;   /* 5000 hits + 5000 misses => balanced */

    /* ---- setup cost: time the one-time table allocation + fault-in ---- */
    uint64_t setup_t0 = tsc_start();   /* counter before setup */
    uint8_t *table = alloc_table();     /* allocate + memset (maps every page) the probe table */
    uint64_t setup_t1 = tsc_end();     /* counter after setup */
    uint64_t setup_cycles = setup_t1 - setup_t0;   /* cycles/ticks spent building the attack buffer */

    int *labels = malloc(TRIALS_PER_RUN * sizeof(int));   /* this run's shuffled hit/miss labels */
    uint64_t *trial_lat = malloc(TRIALS_PER_RUN * sizeof(uint64_t));   /* per-trial latency, kept for median */
    if (!labels || !trial_lat) {   /* allocation guard */
        perror("malloc");   /* report */
        exit(1);             /* error */
    }   /* end if */

    /* ---- memory use: the side-channel working set + harness buffers ---- */
    size_t probe_bytes  = (size_t)NUM_LINES * STRIDE;            /* the meaningful attack footprint */
    size_t labels_bytes = (size_t)TRIALS_PER_RUN * sizeof(int);  /* label bookkeeping */
    size_t lat_bytes    = (size_t)TRIALS_PER_RUN * sizeof(uint64_t);   /* latency bookkeeping */

    printf("# arch=%s threshold=%d polarity=%s trials_per_run=%d runs=%d "   /* one-time metadata line */
           "setup_cycles=%lu probe_bytes=%zu labels_bytes=%zu lat_bytes=%zu\n",
           ARCH_NAME, threshold, hit_above ? "above" : "below", TRIALS_PER_RUN, RUNS,
           (unsigned long)setup_cycles, probe_bytes, labels_bytes, lat_bytes);
    printf("run,trial_id,label,timing,trial_cycles,predicted,correct\n");   /* CSV header (adds per-trial latency) */

    /* ---- warmup: burn throwaway trials so first run isn't measured cold*/
    for (int w = 0; w < MEASURE_WARMUP_ITERS; w++) {   /* throwaway iterations, nothing recorded */
        flush((volatile uint8_t *)&table[0 * STRIDE]);   /* always start from an evicted line, as a trial does */
        fence_full();                                     /* ensure the eviction completed */
        victim(table, 0);                                 /* touch it so the warmup mirrors a hit trial */
        (void)time_flush(&table[0 * STRIDE]);             /* time the probe and throw the result away */
    }   /* end warmup loop */

    for (int run = 0; run < RUNS; run++) {   /* one pass per run */
        srand((unsigned)time(NULL) + (unsigned)run * 7919u + (unsigned)getpid());   /* fresh seed => fresh trials */

        for (int i = 0; i < TRIALS_PER_RUN; i++)              /* build a balanced label array */
            labels[i] = (i < HALF) ? LABEL_HIT : LABEL_MISS;   /* first half hits, second half misses */
        shuffle_order(labels, TRIALS_PER_RUN);                 /* shuffle so hits/misses interleave randomly */

        uint64_t tp = 0, tn = 0, fp = 0, fn = 0;   /* confusion-matrix counters for this run */
        uint64_t lat_sum = 0;                       /* running sum of per-trial latency for the mean */
        uint64_t lat_min = UINT64_MAX, lat_max = 0; /* extremes of per-trial latency */

        struct timespec wall0, wall1;              /* wall-clock bounds for throughput */
        clock_gettime(CLOCK_MONOTONIC, &wall0);   /* start the wall clock */

        for (int t = 0; t < TRIALS_PER_RUN; t++) {   /* run every trial */
            int label = labels[t];                     /* what this trial's ground truth is */

            uint64_t tt0 = tsc_start();                       /* start measuring THIS trial's latency */
            flush((volatile uint8_t *)&table[0 * STRIDE]);   /* always start from an evicted line */
            fence_full();                                     /* ensure the eviction completed */
            if (label == LABEL_HIT)                           /* a hit trial touches the line first */
                victim(table, 0);                              /* ...pulling it into cache */
            uint64_t timing = time_flush(&table[0 * STRIDE]);   /* the measured flush time (the probe) */
            uint64_t tt1 = tsc_end();                          /* stop measuring the trial */
            uint64_t trial_cycles = tt1 - tt0;                 /* full cost of one probe = trial latency */

            lat_sum += trial_cycles;                           /* accumulate for the mean */
            if (trial_cycles < lat_min) lat_min = trial_cycles;   /* track the fastest trial */
            if (trial_cycles > lat_max) lat_max = trial_cycles;   /* track the slowest trial */
            trial_lat[t] = trial_cycles;                       /* stash for the per-run median */

            int pred_hit = hit_above ? ((int)timing > threshold)   /* apply the FROZEN threshold with its polarity */
                                     : ((int)timing < threshold);
            int correct = (pred_hit == (label == LABEL_HIT));       /* did the prediction match ground truth? */

            if (label == LABEL_HIT && pred_hit)       tp++;   /* true positive */
            else if (label == LABEL_MISS && !pred_hit) tn++;  /* true negative */
            else if (label == LABEL_MISS && pred_hit)  fp++;  /* false positive */
            else                                       fn++;  /* false negative (hit predicted as miss) */

            printf("%d,%d,%s,%lu,%lu,%s,%d\n",   /* emit this trial's CSV row (now with trial_cycles) */
                   run, t, (label == LABEL_HIT) ? "hit" : "miss",
                   (unsigned long)timing, (unsigned long)trial_cycles,
                   pred_hit ? "hit" : "miss", correct);
        }   /* end trial loop */

        clock_gettime(CLOCK_MONOTONIC, &wall1);   /* stop the wall clock */
        double wall_sec = (double)(wall1.tv_sec - wall0.tv_sec)      /* whole seconds */
                        + (double)(wall1.tv_nsec - wall0.tv_nsec) / 1e9;   /* plus fractional part */

        uint64_t total = tp + tn + fp + fn;   /* total trials (== TRIALS_PER_RUN) */

        /* ---- accuracy + false positive/negative metrics ---- */
        double accuracy  = 100.0 * (double)(tp + tn) / (double)total;   /* overall correct rate */
        double tpr       = (tp + fn) ? 100.0 * (double)tp / (double)(tp + fn) : 0.0;   /* recall / sensitivity */
        double fpr       = (fp + tn) ? 100.0 * (double)fp / (double)(fp + tn) : 0.0;   /* false-positive rate */
        double fnr       = (tp + fn) ? 100.0 * (double)fn / (double)(tp + fn) : 0.0;   /* false-negative rate */
        double precision = (tp + fp) ? 100.0 * (double)tp / (double)(tp + fp) : 0.0;   /* positive predictive value */
        double f1        = (precision + tpr) ? 2.0 * precision * tpr / (precision + tpr) : 0.0;   /* harmonic mean */

        /* ---- leakage rate (information-theoretic, not just correct-bit counting) ---- */
        double bits_per_trial = mi_bits(tp, tn, fp, fn);   /* mutual information per trial */
        double trials_ps      = total / wall_sec;           /* trial throughput */
        double leak_bps       = bits_per_trial * trials_ps;  /* LEAKAGE RATE: bits of secret per second */

        /* ---- latency: per-trial probe cost, in counter cycles/ticks ---- */
        double lat_mean   = (double)lat_sum / (double)total;        /* mean trial latency */
        uint64_t lat_med  = median_of(trial_lat, TRIALS_PER_RUN);   /* median trial latency (sorts in place) */

        /* ---- memory use: process peak resident set (high-water mark) ---- */
        struct rusage ru;                      /* holds resource-usage counters */
        getrusage(RUSAGE_SELF, &ru);          /* query this process */
        long max_rss_kb = ru.ru_maxrss;        /* peak resident set size, in KiB on Linux */

        fprintf(stderr,   /* per-run summary to stderr so stdout stays pure CSV */
            "run %d summary: "
            "leak_bits_per_trial=%.4f leak_bits_per_sec=%.1f "          /* leakage rate */
            "acc=%.2f%% tpr=%.2f%% fpr=%.2f%% fnr=%.2f%% prec=%.2f%% f1=%.2f "   /* accuracy + FP/FN rates */
            "TP=%lu TN=%lu FP=%lu FN=%lu "                              /* raw false pos/neg counts */
            "lat_cycles(mean/med/min/max)=%.1f/%lu/%lu/%lu "           /* per-trial latency */
            "setup_cycles=%lu "                                         /* one-time setup cost */
            "probe_bytes=%zu max_rss_kb=%ld "                           /* memory use */
            "trials_per_sec=%.1f wall_sec=%.4f\n",                      /* throughput + wall time */
            run,
            bits_per_trial, leak_bps,
            accuracy, tpr, fpr, fnr, precision, f1,
            (unsigned long)tp, (unsigned long)tn, (unsigned long)fp, (unsigned long)fn,
            lat_mean, (unsigned long)lat_med, (unsigned long)lat_min, (unsigned long)lat_max,
            (unsigned long)setup_cycles,
            probe_bytes, max_rss_kb,
            trials_ps, wall_sec);   /* the numbers */
    }   /* end run loop */

    free(trial_lat);   /* cleanup */
    free(labels);      /* cleanup */
    free(table);       /* cleanup */
}   /* end measure */

/* ====================================================================== */
/* main: dispatch on argv[1].                                             */
/* ====================================================================== */
int main(int argc, char **argv)
{   /* begin main */
    if (argc < 2) {   /* no mode supplied */
        fprintf(stderr, "usage: %s calibrate\n", argv[0]);   /* show the calibrate form */
        fprintf(stderr, "       %s measure <threshold> <below|above>\n", argv[0]);   /* and the measure form */
        return 1;   /* error */
    }   /* end if */

    if (strcmp(argv[1], "calibrate") == 0) {   /* calibration mode */
        calibrate();   /* collect samples + print a frozen threshold suggestion */
        return 0;       /* done */
    }   /* end if */

    if (strcmp(argv[1], "measure") == 0) {   /* measurement mode */
        if (argc < 4) {   /* measure needs the frozen threshold and its polarity */
            fprintf(stderr, "usage: %s measure <threshold> <below|above>\n", argv[0]);   /* remind the caller */
            return 1;   /* error */
        }   /* end if */
        int threshold = (int)strtol(argv[2], NULL, 10);   /* the FROZEN threshold from calibration */
        int hit_above;   /* polarity: 1 => predict hit when timing > threshold */
        if (strcmp(argv[3], "above") == 0)      hit_above = 1;   /* "above" polarity */
        else if (strcmp(argv[3], "below") == 0) hit_above = 0;   /* "below" polarity */
        else {   /* anything else is a usage error */
            fprintf(stderr, "polarity must be 'below' or 'above'\n");   /* explain */
            return 1;   /* error */
        }   /* end else */
        measure(threshold, hit_above);   /* run the 10x10000 balanced sweep */
        return 0;   /* done */
    }   /* end if */

    fprintf(stderr, "unknown mode: %s\n", argv[1]);   /* did not match calibrate/measure */
    return 1;   /* error */
}   /* end main */
