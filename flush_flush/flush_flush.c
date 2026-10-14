#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    //for time() to seed rand() in main
#include <unistd.h>   // getpid()

//tunable constants
#define CACHE_LINE   64      /* bytes; probes must sit on separate lines. WHY 64? */
#define NUM_SAMPLES  10000  /* repetitions for the calibration histogram */
#define THRESHOLD 256 //placeholder until tested

/* rdtsc_start(): read the timestamp counter to OPEN a timing window.
 * Return: the 64-bit counter value (combine the two 32-bit halves). */
static inline uint64_t rdtsc_start(void)
{
    uint32_t lo; //will hold low the 32 bits of 64 bit counter
    uint32_t hi; //will hold the high 32 bits of 64 bit counter
    asm volatile (
        "lfence\n\t"   /* wait for all EARLIER loads to finish, so work from
                          before the timer can't bleed into the measurement */
        "rdtsc\n\t"    /* read the 64-bit Time-Stamp Counter:
                          hardware puts low 32 bits in EAX, high 32 bits in EDX */
        : "=a"(lo),    /* output: tie EAX into lo   ("a" = eax/rax register) */
          "=d"(hi)     /* output: tie EDX into hi   ("d" = edx/rdx register) */
        :              /* no inputs */
        : "memory"     /* clobber: compiler must not move memory ops across this */
    );
    return ((uint64_t)hi << 32) | lo;
    /* widen hi to 64 bits FIRST, then shift up by 32, then OR in lo.
       widening before the shift is what stops the high bits falling off. */
}

// rdtsc_end(): read the TSC to CLOSE the window.
static inline uint64_t rdtsc_end(void)
{
    uint32_t lo;   /* low  32 bits of the counter */
    uint32_t hi;   /* high 32 bits of the counter */

    asm volatile (
        "rdtscp\n\t"   //read TSC but wait for all earlier instructions to retire first
        "lfence\n\t"   //stop later instructions from sneaking back before the clock read. On the end side of fence goes after.
        : "=a"(lo),    //eas us ki
          "=d"(hi)     //edx is high
        :              
        : "rcx", "memory"  //rdtscp writes ecx so we must declare rcx clobbered
    );

    return ((uint64_t)hi << 32) | lo;   //same as rdtsc_start
}

// flush(addr): evict ONE line (the one containing addr) from the caches.
static inline void flush(void *addr)
{
    asm volatile (
        "clflush (%0)\n\t"   //evict the cache line containing addr from all levels. %0 means memory at address in oeprand 0.
        :                    //no outputs
        : "r"(addr)          //input 0: addr, placed in any general register. r = any register the compiler picks
        : "memory"           //this changes memory state the compiler can't see so dont reorder anything across it
    );
}

// clflush is weakly ordered on its own, so fence around it or the
// timing read can slide in before the flush is actually done
static inline void mfence(void)    // no args, no return
{   // start of function body
    asm volatile ("mfence\n\t" : : : "memory");   // full fence, nothing to wire up
}   // end mfence

static uint64_t time_flush(volatile uint8_t *addr)    // addr: line to flush+time, returns cycles elapsed
{   // start of function body
    uint64_t t0, t1;        // start/end tsc values

    mfence();                // make sure nothing earlier is still in flight
    t0 = rdtsc_start();      // start the clock

    flush((void *)addr);    // do the actual flush
    mfence();                 // and wait for it to land

    t1 = rdtsc_end();        // stop the clock
    return t1 - t0;           // cycles the flush took -- small = was cached, big = wasn't
}   // end time_flush



#define NUM_SECRETS 16    // how many candidate lines we're guessing between
#define STRIDE 4096        // one page apart so each probe line is its own page

static int probe_order[NUM_SECRETS];             // current shuffled scan order
static void init_probe_order(void);              // fills/shuffles probe_order, defined near the bottom
static void shuffle_order(int *order, int n);     // generic shuffle, also defined near the bottom

static void victim(volatile uint8_t *table, int secret)    // table: shared buffer, secret: which line to touch
{   // start of function body
    (void)table[secret * STRIDE];   // just touch the one line, that's the whole "secret access"
}   // end victim

// sorts values in place and returns the middle element. for n=10000 this
// isn't a textbook median (even n, no averaging the two middle values) but
// it's close enough for picking a threshold
static uint64_t median_of(uint64_t *values, int n)    // values gets sorted as a side effect
{   // start of function body
    for (int i = 1; i < n; i++) {           // insertion sort, i walks the unsorted part
        uint64_t key = values[i];            // value being inserted
        int j = i - 1;                        // walk left through the sorted part
        while (j >= 0 && values[j] > key) {  // shift anything bigger than key to the right
            values[j + 1] = values[j];          // slide it over
            j--;                                  // keep walking left
        }   // end while
        values[j + 1] = key;    // key's final spot
    }   // end for
    return values[n / 2];   // middle-ish value once everything's sorted
}   // end median_of

// collects hit/miss cycle samples using the same table layout as the real
// attack (NUM_SECRETS+1 lines, flush everything, shuffle the probe order)
// instead of a single standalone page. tried calibrating against one page
// early on and the threshold it gave was useless against the real table --
// every probe read above it. table shape actually matters here.
static void calibrate(void)    // prints "hit N"/"miss N" lines plus a suggested threshold
{   // start of function body
    srand((unsigned)time(NULL));   // seed once for this run

    uint8_t *table;   // the probe table itself
    if (posix_memalign((void **)&table, 4096, (size_t)(NUM_SECRETS + 1) * STRIDE) != 0) {   // grab aligned memory
        perror("posix_memalign");   // couldn't get the memory
        exit(1);                     // nothing to do without it
    }   // end if
    memset(table, 1, (size_t)(NUM_SECRETS + 1) * STRIDE);   // touch every page so it's actually mapped in

    int flush_order[NUM_SECRETS + 1];   // shuffled order for the flush-all pass

    uint64_t *hit_samples  = malloc(NUM_SAMPLES * sizeof(uint64_t));   // every hit sample this run
    uint64_t *miss_samples = malloc(NUM_SAMPLES * sizeof(uint64_t));   // every miss sample this run
    if (!hit_samples || !miss_samples) {   // either alloc failing means no threshold this time
        perror("malloc");    // ran out of memory
        exit(1);              // can't compute a threshold without the buffers
    }   // end if

    // run the loop a few thousand times before actually recording anything.
    // the first chunk of samples read noticeably slower than steady state --
    // probably clock ramping up, maybe some cache/TLB warmup too -- and that
    // drags the median around if it's not burned off first. 500 wasn't
    // enough, the median kept drifting down past that, so bumped it up.
    #define CALIB_WARMUP_ITERS 3000   // how many throwaway iterations before recording starts
    for (int w = 0; w < CALIB_WARMUP_ITERS; w++) {    // throwaway iterations, nothing recorded
        init_probe_order();                             // fresh scan order
        shuffle_order(flush_order, NUM_SECRETS + 1);    // fresh flush order

        for (int j = 0; j <= NUM_SECRETS; j++)          // flush everything first
            flush((void *)&table[flush_order[j] * STRIDE]);   // evict this line

        int secret = rand() % NUM_SECRETS;   // pretend-victim line
        victim(table, secret);                 // touch it

        for (int k = 0; k < NUM_SECRETS; k++)          // scan everything, throw the timings away
            (void)time_flush(&table[probe_order[k] * STRIDE]);   // time it and discard
    }   // end warmup loop

    for (int i = 0; i < NUM_SAMPLES; i++) {   // the real recorded iterations
        init_probe_order();                          // fresh scan order, same as a real trial would use
        shuffle_order(flush_order, NUM_SECRETS + 1);  // fresh flush order too

        for (int j = 0; j <= NUM_SECRETS; j++)        // flush all 17 lines before touching anything
            flush((void *)&table[flush_order[j] * STRIDE]);   // evict this line

        int secret = rand() % NUM_SECRETS;   // this round's victim line
        victim(table, secret);                 // one touch, like a real attack would see

        // pick one of the 15 untouched lines at random to be this round's
        // miss sample, instead of always using the same position
        int miss_pick = rand() % (NUM_SECRETS - 1);   // which untouched line to keep
        int miss_seen = 0;                               // counts untouched lines as we pass them
        uint64_t hit_cycles = 0, miss_cycles = 0;        // this round's hit sample and chosen miss sample

        for (int k = 0; k < NUM_SECRETS; k++) {   // scan every candidate
            int idx = probe_order[k];                // which line this step is
            uint64_t t = time_flush(&table[idx * STRIDE]);   // how long the flush took

            if (idx == secret) {   // this is the line we touched
                hit_cycles = t;   // keep it as the hit sample
            } else {   // untouched line
                if (miss_seen == miss_pick)   // is this the one we're keeping
                    miss_cycles = t;   // this is the untouched line we're keeping
                miss_seen++;             // move on to the next untouched candidate
            }   // end else
        }   // end for k

        printf("hit %lu\n", hit_cycles);     // dump this round's hit sample
        printf("miss %lu\n", miss_cycles);   // and its miss sample

        hit_samples[i]  = hit_cycles;    // keep it around for the median/threshold math below
        miss_samples[i] = miss_cycles;   // same for the miss side
    }   // end for i

    free(table);   // done with the table

    // median_of sorts in place as a side effect, which is also handy for
    // the misclassification count below
    uint64_t hit_median  = median_of(hit_samples, NUM_SAMPLES);     // typical hit timing
    uint64_t miss_median = median_of(miss_samples, NUM_SAMPLES);    // typical miss timing
    int threshold = (int)((hit_median + miss_median) / 2);           // split the difference

    // on this machine a cached line actually flushes FASTER than an
    // uncached one, backwards from what most writeups assume. so "hit"
    // means below the threshold here, not above.
    uint32_t false_miss = 0;   // hit samples that would get called a miss at this threshold
    for (int i = 0; i < NUM_SAMPLES; i++)   // walk every hit sample
        if ((int)hit_samples[i] > threshold)   // reads above threshold, wrongly called miss
            false_miss++;   // tally it

    uint32_t false_hit = 0;    // miss samples that would get called a hit at this threshold
    for (int i = 0; i < NUM_SAMPLES; i++)   // walk every miss sample
        if ((int)miss_samples[i] <= threshold)   // reads at/below threshold, wrongly called hit
            false_hit++;   // tally it

    uint32_t misclassified = false_miss + false_hit;   // total errors, both directions

    fprintf(stderr,   // summary goes to stderr
        "calibration summary: suggested THRESHOLD=%d (midpoint of hit_median=%lu, miss_median=%lu) "   // first half of the message
        "misclassified=%u/%d (%.2f%%)\n",   // second half, these concat into one string
        threshold, hit_median, miss_median, misclassified, 2 * NUM_SAMPLES,   // the actual values
        100.0 * misclassified / (2 * NUM_SAMPLES));   // and the percentage

    free(hit_samples);    // cleanup
    free(miss_samples);   // cleanup
}   // end calibrate

static int recover_one(volatile uint8_t *table, int secret)    // returns guessed index or -1
{   // start of function body
    for (int i = 0; i < NUM_SECRETS; i++)   // flush every candidate so we start uncached
        flush((void *)&table[i * STRIDE]);   // evict this one

    victim(table, secret);   // let the "victim" touch its line

    // whichever line flushes faster than THRESHOLD is the one the victim touched
    for (int i = 0; i < NUM_SECRETS; i++) {   // walk the shuffled scan order
        int idx = probe_order[i];                        // which candidate this step checks
        uint64_t t = time_flush(&table[idx * STRIDE]);   // how fast it flushed
        if (t < THRESHOLD)   // fast enough to be the touched line
            return idx;   // found it
    }   // end for
    return -1;   // nothing crossed the threshold
}   // end recover_one

// full measurement sweep for the writeup. table has 17 lines: 16 real
// candidates plus one the victim never touches, used as a control to catch
// false positives. the control gets probed at a random point in the scan
// instead of always last so it isn't biased by being the final flush every
// time. everything's timed in cycles via rdtsc for the per-trial latency
// numbers.
static int recover_one_measure(volatile uint8_t *table,   // shared buffer, NUM_SECRETS+1 lines
                                int secret,                // line the victim should touch, ignored if run_victim is 0
                                int run_victim,             // 1 = real trial, 0 = no-victim control trial
                                uint64_t *probe_times,      // out: one cycle count per real line
                                uint64_t *control_time,     // out: cycle count for the unrelated line
                                int *control_hit,           // out: 1 if the control line read as a hit
                                uint64_t *trial_cycles)     // out: cycles for the whole trial
{   // start of function body
    uint64_t t0 = rdtsc_start();   // start timing the whole trial

    int flush_order[NUM_SECRETS + 1];           // shuffled order for the flush-all pass
    shuffle_order(flush_order, NUM_SECRETS + 1);   // fresh every trial
    for (int i = 0; i <= NUM_SECRETS; i++)         // flush all 16 candidates plus the control line
        flush((void *)&table[flush_order[i] * STRIDE]);   // evict this one

    if (run_victim)          // control trials skip this
        victim(table, secret);   // touch the secret line

    int probe_scan[NUM_SECRETS + 1];           // shuffled order for the probe pass
    shuffle_order(probe_scan, NUM_SECRETS + 1);   // control's position moves around trial to trial

    int guess = -1;   // first candidate that reads as a hit, or -1 if none do
    for (int i = 0; i <= NUM_SECRETS; i++) {   // walk the shuffled scan
        int idx = probe_scan[i];                        // which line this step checks
        uint64_t t = time_flush(&table[idx * STRIDE]);   // how fast it flushed

        if (idx == NUM_SECRETS) {   // this step is the control line
            *control_time = t;                 // record the control line's own timing
            *control_hit  = (t < THRESHOLD);   // did it wrongly read as a hit?
        } else {   // this step is a real candidate
            probe_times[idx] = t;             // keep every candidate's timing for the csv dump
            if (t < THRESHOLD && guess == -1)   // first one crossing threshold wins
                guess = idx;   // lock in the first one that crosses
        }   // end else
    }   // end for

    uint64_t t1 = rdtsc_end();    // stop timing the trial
    *trial_cycles = t1 - t0;      // hand back how long it took
    return guess;                  // -1 if nothing crossed
}   // end recover_one_measure

static void print_probe_times(const uint64_t *probe_times)    // dumps all NUM_SECRETS timings, semicolon separated
{   // start of function body
    for (int i = 0; i < NUM_SECRETS; i++)   // dump each candidate's timing, semicolon separated
        printf("%lu%s", probe_times[i], (i + 1 < NUM_SECRETS) ? ";" : "");   // no trailing separator on the last one
}   // end print_probe_times

// 10 runs of 10,000 trials each, plus 1000 no-victim control trials per run,
// dumped as CSV on stdout with the per-run summary going to stderr so they
// don't get mixed together (redirect with `./ff measure > data.csv`)
static void measure(void)    // drives the full 10-run sweep
{   // start of function body
    const int RUNS             = 10;       // how many independent runs
    const int TRIALS_PER_RUN   = 10000;    // real trials per run
    const int CONTROL_TRIALS   = 1000;     // no-victim trials per run

    uint64_t setup_t0 = rdtsc_start();   // time the one-time table setup
    uint8_t *table;   // shared probe table for the whole sweep
    if (posix_memalign((void **)&table, 4096, (size_t)(NUM_SECRETS + 1) * STRIDE) != 0) {   // allocate it
        perror("posix_memalign");   // couldn't allocate the table
        exit(1);                     // nothing else to do
    }   // end if
    memset(table, 1, (size_t)(NUM_SECRETS + 1) * STRIDE);   // map every page in up front
    uint64_t setup_t1 = rdtsc_end();                 // stop timing setup
    uint64_t setup_cycles = setup_t1 - setup_t0;     // how long setup took

    size_t mem_bytes = (size_t)(NUM_SECRETS + 1) * STRIDE;   // table size, for the header row

    printf("# setup_cycles=%lu mem_bytes=%zu threshold=%d\n", setup_cycles, mem_bytes, THRESHOLD);   // one-time metadata line
    printf("run,trial_type,trial_id,secret,guess,correct,control_hit,trial_cycles,probe_times\n");    // csv header

    for (int run = 0; run < RUNS; run++) {   // one pass of this loop per run
        srand((unsigned)time(NULL) + (unsigned)run * 7919u + (unsigned)getpid());   // fresh seed each run

        int correct = 0, false_negatives = 0;   // tallies for this run

        struct timespec wall0, wall1;              // wall clock bounds for the real trials
        clock_gettime(CLOCK_MONOTONIC, &wall0);   // start the wall clock

        for (int t = 0; t < TRIALS_PER_RUN; t++) {   // the real trials
            int secret = rand() % NUM_SECRETS;   // this trial's victim line
            uint64_t probe_times[NUM_SECRETS], control_time, trial_cycles;   // out params from recover_one_measure
            int control_hit;                                                   // same
            int guess = recover_one_measure(table, secret, 1,   // run_victim=1, real trial
                                             probe_times, &control_time, &control_hit, &trial_cycles);   // run the trial
            int correct_flag = (guess == secret);   // did we get it right
            correct += correct_flag;                  // tally it
            if (guess == -1)   // nothing ever crossed threshold
                false_negatives++;   // touched a line but never detected a hit

            printf("%d,real,%d,%d,%d,%d,%d,%lu,", run, t, secret, guess, correct_flag,   // identifying + outcome fields
                   control_hit, trial_cycles);   // identifying + outcome fields for this row
            print_probe_times(probe_times);        // all 16 candidate timings
            printf("\n");                              // end the row
        }   // end real trial loop

        clock_gettime(CLOCK_MONOTONIC, &wall1);   // stop the wall clock
        double wall_sec = (double)(wall1.tv_sec - wall0.tv_sec)      // seconds spent on real trials
                         + (double)(wall1.tv_nsec - wall0.tv_nsec) / 1e9;   // plus the fractional part

        int false_positives = 0, control_line_violations = 0;   // tallies for the control trials
        for (int t = 0; t < CONTROL_TRIALS; t++) {   // the no-victim control trials
            uint64_t probe_times[NUM_SECRETS], control_time, trial_cycles;   // out params again
            int control_hit;                                                   // same
            int guess = recover_one_measure(table, -1, 0,   // run_victim=0, no secret for a control trial
                                             probe_times, &control_time, &control_hit, &trial_cycles);   // run the control trial
            if (guess != -1)   // claimed a hit even though nothing was touched
                false_positives++;   // claimed a hit with nothing touched
            if (control_hit)   // the unrelated line itself read as a hit
                control_line_violations++;   // the control line itself misread as a hit

            printf("%d,control,%d,%d,%d,%d,%d,%lu,", run, t, -1, guess, 0,   // secret is n/a, correct is always 0
                   control_hit, trial_cycles);   // same row format, secret is n/a here
            print_probe_times(probe_times);   // same timings dump
            printf("\n");   // end the row
        }   // end control trial loop

        double accuracy     = 100.0 * correct / TRIALS_PER_RUN;              // % of real trials recovered right
        double fn_rate       = 100.0 * false_negatives / TRIALS_PER_RUN;      // % with no hit detected at all
        double fp_rate        = 100.0 * false_positives / CONTROL_TRIALS;      // % of control trials that still "hit"
        double trials_per_sec = TRIALS_PER_RUN / wall_sec;                     // throughput
        double bits_per_sec   = (correct * 4.0) / wall_sec;   // log2(16) bits per correct guess

        fprintf(stderr,   // per-run summary goes to stderr, stdout stays pure csv
            "run %d summary: accuracy=%.2f%% fn_rate=%.2f%% fp_rate=%.2f%% "   // first chunk of the message
            "trials_per_sec=%.1f bits_per_sec=%.1f control_line_violations=%d/%d "   // middle chunk
            "setup_cycles=%lu mem_bytes=%zu wall_sec=%.4f\n",   // last chunk, all concat together
            run, accuracy, fn_rate, fp_rate, trials_per_sec, bits_per_sec,   // the main numbers
            control_line_violations, CONTROL_TRIALS, setup_cycles, mem_bytes, wall_sec);   // the rest
    }   // end run loop

    free(table);   // all done
}   // end measure

int main(int argc, char **argv)    // argv[1] picks the mode
{   // start of function body
    if (argc < 2) {   // no mode given
        fprintf(stderr, "usage: %s calibrate|attack|measure\n", argv[0]);   // print usage
        return 1;                                                              // bail
    }   // end if

    if (strcmp(argv[1], "calibrate") == 0) {   // mode is calibrate
        calibrate();     // runs the calibration sweep, prints hit/miss samples + a suggested threshold
        return 0;          // done
    }   // end if

    if (strcmp(argv[1], "attack") == 0) {   // mode is attack
        uint8_t *table;   // probe table for this run
        if (posix_memalign((void **)&table, 4096, (size_t)NUM_SECRETS * STRIDE) != 0) {   // allocate it
            perror("posix_memalign");   // couldn't allocate it
            return 1;                     // nothing allocated yet, just bail
        }   // end if
        memset(table, 1, (size_t)NUM_SECRETS * STRIDE);   // map it in

        srand((unsigned)time(NULL));   // seed for this run
        init_probe_order();              // build the first scan order

        int trials = 1000;   // how many secrets to try
        int correct = 0;       // how many we got right

        for (int t = 0; t < trials; t++) {   // run the trials
            int secret = rand() % NUM_SECRETS;       // pick a secret
            int guess  = recover_one(table, secret);   // try to recover it
            if (guess == secret)   // got it right
                correct++;   // got it
        }   // end for

        printf("accuracy: %d/%d = %.1f%%\n", correct, trials, 100.0 * correct / trials);   // final tally

        free(table);   // cleanup
        return 0;        // done
    }   // end if

    if (strcmp(argv[1], "measure") == 0) {   // mode is measure
        measure();    // runs the full 10x10000 sweep described above
        return 0;       // done
    }   // end if

    fprintf(stderr, "unknown mode: %s\n", argv[1]);   // didn't match any of the three modes
    return 1;                                            // bail
}   // end main

// fisher-yates. used both for the probe scan order and the flush-all
// preamble so neither one walks the table in address order
static void shuffle_order(int *order, int n)    // order: array to fill, n: how many slots
{   // start of function body
    for (int i = 0; i < n; i++)   // start off as identity
        order[i] = i;   // identity mapping before the shuffle

    for (int i = n - 1; i > 0; i--) {   // walk backwards, swapping each slot with an earlier random one
        int j = rand() % (i + 1);        // pick the swap partner
        int tmp = order[i];                // standard swap
        order[i] = order[j];               // first half of the swap
        order[j] = tmp;                     // second half of the swap
    }   // end for
}   // end shuffle_order

static void init_probe_order(void)    // fills/shuffles the file-scope probe_order array
{   // start of function body
    shuffle_order(probe_order, NUM_SECRETS);   // just hands off to the generic shuffle
}   // end init_probe_order
