// dmp_test.c
// Data Memory-dependent Prefetcher (DMP) detection test for Intel x86 (Linux).
//
// Idea (array-of-pointers test): a DMP sees a load return a pointer-like value
// and prefetches *value even though the program never dereferences it.
//   1. arr[] holds pointers to cache lines in data[].
//   2. We flush everything, then only READ arr[0..K-1] (training).
//   3. We time loads of data[] lines that the program never touched.
//   4. A fast load means something prefetched that line from the pointer value.
// Matched controls (see COND_* below) rule out competing explanations.
//
// Build: gcc -O1 -o dmp_test dmp_test.c
// Usage: ./dmp_test <run_id> <trials> <cpu>   > raw.csv   (CSV header is always printed)

#define _GNU_SOURCE                       // needed for sched_setaffinity / CPU_SET
#include <stdio.h>                        // printf, fprintf
#include <stdlib.h>                       // aligned_alloc, atoi, exit
#include <stdint.h>                       // fixed-width integer types
#include <string.h>                       // memset, memcpy
#include <unistd.h>                       // read, close, syscall
#include <sched.h>                        // sched_setaffinity
#include <sys/syscall.h>                  // SYS_perf_event_open
#include <sys/ioctl.h>                    // ioctl for perf counters
#include <linux/perf_event.h>             // perf_event_attr and constants
#include <x86intrin.h>                    // __rdtscp, _mm_clflush, fences

#define LINE 64                           // cache line size in bytes
#define N 256                             // number of pointer entries in arr[]
#define K 64                              // entries the program really reads (training length)
#define M 32                              // number of data lines probed per trial
#define WARMUP 100                        // trials discarded at the start of each process
#define NCOND 6                           // number of conditions (test + controls)

// Condition ids (written to the CSV as integers; analyze_dmp.py maps them to names).
enum { COND_TEST = 0,                     // arr holds valid pointers, train, probe never-touched targets
       COND_NONPTR = 1,                   // control: same arr traffic but values are NOT pointer-like
       COND_NOTRAIN = 2,                  // control: valid pointers but arr is never read (cold baseline)
       COND_NEIGHBOR = 3,                 // control: probe the adjacent line of each target (spatial prefetch)
       COND_SEQ = 4,                      // control: targets are laid out sequentially (stride-friendly)
       COND_DEREF = 5 };                  // control: program really dereferences (cached upper bound)

static uint64_t rng_state = 88172645463325252ULL;  // xorshift64 state for shuffling

// Return a pseudo-random 64-bit number (xorshift64).
static inline uint64_t rng(void) {
    rng_state ^= rng_state << 13;         // xorshift step 1
    rng_state ^= rng_state >> 7;          // xorshift step 2
    rng_state ^= rng_state << 17;         // xorshift step 3
    return rng_state;                     // hand back the new state as the random value
}

// Time one load of *p in TSC cycles, serialized so the load is isolated.
static inline uint64_t time_load(volatile uint8_t *p) {
    unsigned aux;                         // required output of rdtscp (core id), unused
    uint64_t t0, t1;                      // start and end timestamps
    _mm_mfence();                         // drain earlier memory operations
    _mm_lfence();                         // stop speculation past this point
    t0 = __rdtscp(&aux);                  // read start timestamp
    _mm_lfence();                         // make sure the timestamp is taken before the load
    (void)*p;                             // the load we are timing
    _mm_lfence();                         // wait for the load to complete
    t1 = __rdtscp(&aux);                  // read end timestamp
    _mm_lfence();                         // keep later code from moving before the end timestamp
    return t1 - t0;                       // elapsed cycles for the load
}

// Open one user-space perf counter on this thread; returns fd or -1 if not permitted.
static int open_counter(uint32_t type, uint64_t config) {
    struct perf_event_attr pe;            // perf event description
    memset(&pe, 0, sizeof(pe));           // zero all fields first
    pe.type = type;                       // PERF_TYPE_RAW or PERF_TYPE_HARDWARE
    pe.size = sizeof(pe);                 // struct size required by the kernel
    pe.config = config;                   // which event to count
    pe.disabled = 0;                      // start counting immediately
    pe.exclude_kernel = 1;                // count user-space only (works with paranoid <= 2)
    pe.exclude_hv = 1;                    // exclude hypervisor
    return (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);  // this thread, any cpu
}

// Read a counter's current value; returns 0 if the counter is unavailable.
static inline uint64_t read_counter(int fd) {
    uint64_t v = 0;                       // counter value read from the kernel
    if (fd < 0) return 0;                 // counter not opened, report 0
    if (read(fd, &v, sizeof(v)) != sizeof(v)) return 0;  // failed read, report 0
    return v;                             // raw counter value
}

int main(int argc, char **argv) {
    if (argc != 4) {                      // need exactly three arguments
        fprintf(stderr, "usage: %s <run_id> <trials> <cpu>\n", argv[0]);  // tell the user
        return 1;                         // error exit
    }
    int run_id = atoi(argv[1]);           // which independent process run this is (1-based)
    int trials = atoi(argv[2]);           // measured trials per condition
    int cpu = atoi(argv[3]);              // core to pin to

    cpu_set_t set;                        // CPU affinity mask
    CPU_ZERO(&set);                       // start with an empty mask
    CPU_SET(cpu, &set);                   // allow only the chosen core
    if (sched_setaffinity(0, sizeof(set), &set) != 0)  // pin this process
        fprintf(stderr, "warning: could not pin to cpu %d\n", cpu);  // continue unpinned

    rng_state ^= (uint64_t)run_id * 0x9E3779B97F4A7C15ULL ^ (uint64_t)__rdtsc();  // per-run seed

    int fd_l1 = open_counter(PERF_TYPE_RAW, 0x08D1);  // MEM_LOAD_RETIRED.L1_MISS (event D1, umask 08)
    int fd_llc = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);  // LLC misses
    if (fd_l1 < 0 || fd_llc < 0)          // counters are optional
        fprintf(stderr, "warning: perf counters unavailable, l1/llc columns will be 0\n");

    // arr variants: valid pointers (random order), non-pointers, valid pointers (sequential order).
    uint64_t *arr_valid = aligned_alloc(4096, N * sizeof(uint64_t));   // pointer array for TEST/NOTRAIN/NEIGHBOR/DEREF
    uint64_t *arr_nonptr = aligned_alloc(4096, N * sizeof(uint64_t));  // same values with non-canonical high bits
    uint64_t *arr_seq = aligned_alloc(4096, N * sizeof(uint64_t));     // pointers to sequentially placed lines
    // data holds 2 lines per slot: even line = pointer target, odd line = never pointed to (neighbor).
    uint8_t *data = aligned_alloc(4096, (size_t)N * 2 * LINE);         // target region
    memset(data, 1, (size_t)N * 2 * LINE);  // touch every page so it is mapped
    memset(arr_valid, 0, N * sizeof(uint64_t));   // map arr_valid
    memset(arr_nonptr, 0, N * sizeof(uint64_t));  // map arr_nonptr
    memset(arr_seq, 0, N * sizeof(uint64_t));     // map arr_seq

    int perm[N];                          // random slot order for each trial (kept cached, not flushed)
    int order[NCOND];                     // randomized condition order within a trial round
    volatile uint64_t sink = 0;           // keeps training loads from being optimized away

    printf("run,trial,cond,j,cycles,l1_miss,llc_miss\n");  // CSV header (run_dmp.sh strips duplicates when merging)

    for (int t = -WARMUP; t < trials; t++) {  // negative trials are warm-up and not printed
        for (int c = 0; c < NCOND; c++) order[c] = c;      // start with the identity order
        for (int c = NCOND - 1; c > 0; c--) {              // Fisher-Yates shuffle of conditions
            int r = (int)(rng() % (uint64_t)(c + 1));      // pick a random index <= c
            int tmp = order[c];           // swap order[c] and order[r]
            order[c] = order[r];          // move random choice into position c
            order[r] = tmp;               // finish the swap
        }
        for (int oi = 0; oi < NCOND; oi++) {               // run every condition once this round
            int cond = order[oi];         // the condition for this measurement
            for (int i = 0; i < N; i++) perm[i] = i;       // reset permutation to identity
            for (int i = N - 1; i > 0; i--) {              // Fisher-Yates shuffle of target slots
                int r = (int)(rng() % (uint64_t)(i + 1));  // random index <= i
                int tmp = perm[i];        // swap perm[i] and perm[r]
                perm[i] = perm[r];        // move random slot into position i
                perm[r] = tmp;            // finish the swap
            }
            for (int i = 0; i < N; i++) {                  // fill the three arr variants
                uint64_t p = (uint64_t)(data + (size_t)perm[i] * 2 * LINE);  // random-order target pointer
                arr_valid[i] = p;         // valid pointer to target line
                arr_nonptr[i] = p ^ 0xFFFF000000000000ULL; // same bits but non-canonical (not pointer-like)
                arr_seq[i] = (uint64_t)(data + (size_t)i * 2 * LINE);  // sequential target pointer
            }
            for (int i = 0; i < N; i++) _mm_clflush(&arr_valid[i]);   // evict arr_valid from all caches
            for (int i = 0; i < N; i++) _mm_clflush(&arr_nonptr[i]);  // evict arr_nonptr
            for (int i = 0; i < N; i++) _mm_clflush(&arr_seq[i]);     // evict arr_seq
            for (size_t b = 0; b < (size_t)N * 2 * LINE; b += LINE) _mm_clflush(data + b);  // evict all targets
            _mm_mfence();                 // make sure all flushes are complete

            uint64_t *train = arr_valid;  // which array the training loop reads
            if (cond == COND_NONPTR) train = arr_nonptr;   // control: non-pointer values
            if (cond == COND_SEQ) train = arr_seq;         // control: sequential targets

            if (cond != COND_NOTRAIN) {   // NOTRAIN skips the training reads entirely
                for (int i = 0; i < K; i++) {              // sequentially read the first K entries
                    uint64_t v = train[i];                 // load the (possibly pointer-like) value
                    sink += v & 1;        // use the value without dereferencing it
                    if (cond == COND_DEREF)                // upper bound: really dereference the pointer
                        sink += *(volatile uint8_t *)v;    // this pulls the target line into cache
                }
            }

            uint64_t l1_0 = read_counter(fd_l1);           // L1 miss count before probing
            uint64_t llc_0 = read_counter(fd_llc);         // LLC miss count before probing
            uint64_t cyc[M];              // probe latencies for this trial
            for (int m = 0; m < M; m++) { // probe M target lines
                int j = (cond == COND_DEREF) ? m : K + m;  // DEREF probes lines it read; others probe unread entries
                size_t line = (size_t)(cond == COND_SEQ ? j : perm[j]) * 2;  // even line = pointer target
                if (cond == COND_NEIGHBOR) line += 1;      // probe the odd neighbor nobody points to
                cyc[m] = time_load(data + line * LINE);    // timed load of the probed line
            }
            uint64_t l1_1 = read_counter(fd_l1);           // L1 miss count after probing
            uint64_t llc_1 = read_counter(fd_llc);         // LLC miss count after probing

            if (t >= 0) {                 // only print measured (non-warm-up) trials
                for (int m = 0; m < M; m++)                // one CSV row per probed line
                    printf("%d,%d,%d,%d,%llu,%llu,%llu\n", run_id, t, cond, m,
                           (unsigned long long)cyc[m],     // probe latency in cycles
                           (unsigned long long)(l1_1 - l1_0),    // L1 misses during probe window
                           (unsigned long long)(llc_1 - llc_0)); // LLC misses during probe window
            }
        }
    }
    if (sink == 0xdeadbeefULL) fprintf(stderr, "sink\n");  // never true; stops sink being optimized out
    if (fd_l1 >= 0) close(fd_l1);         // release the L1 counter
    if (fd_llc >= 0) close(fd_llc);       // release the LLC counter
    return 0;                             // success
}
