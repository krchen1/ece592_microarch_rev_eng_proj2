#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    //for time() to seed rand() in main

//tunable constants
#define CACHE_LINE   64      /* bytes; probes must sit on separate lines. WHY 64? */
#define NUM_SAMPLES  100000  /* repetitions for the calibration histogram */
#define THRESHOLD 200 //placeholder until tested

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

//timed reload
static uint64_t time_reload(volatile uint8_t *addr)
{
    uint64_t t0, t1;        //start and end timestamps
    volatile uint8_t val;   //holds the loaded byte so it cant vanish

    t0 = rdtsc_start();     //open the window, fence, then read clock

    val = *addr;            //one load of the line (event im measuring)

    t1 = rdtsc_end();       //close the timing window

    (void)val;              //mark val used

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
        uint64_t cycles = time_reload(probe);
        printf("hit %lu\n", cycles);     //dump raw into stats
    }

    //flush then time it (miss loop)
    for (int i = 0; i < NUM_SAMPLES; i++) {
        flush((void *)probe);            //evict
        uint64_t cycles = time_reload(probe);
        printf("miss %lu\n", cycles);
    }

    free(buf);
}


#define NUM_SECRETS 16  //secret
#define STRIDE 4096 //spacing between probe lines (one page)

static int probe_order[NUM_SECRETS];   //shuffled reload order; defeats the prefetcher
static void init_probe_order(void);    //fills + shuffles probe_order (defined below main)

static void victim(volatile uint8_t *table, int secret)
{
    (void)table[secret * STRIDE];   //data dependent load which line is touched depends on secret
}

//atatack loop
static int recover_one(volatile uint8_t *table, int secret)
{
    //flush every candidate line so we start uncached
    for (int i = 0; i < NUM_SECRETS; i++)
        flush((void *)&table[i * STRIDE]);

    //let victim run.
    victim(table, secret);

    // reload every candidate and time each. One below threshold is the line the victim touched. 
    for (int i = 0; i < NUM_SECRETS; i++) {
        int idx = probe_order[i];                     //shuffled index
        uint64_t t = time_reload(&table[idx * STRIDE]);
        if (t < THRESHOLD)
            return idx;                               //hit = secret
    }
    return -1;                                        //no hit detected

}



int main(int argc, char **argv)
{
    //need a mode: ./fr calibrate   or   ./fr attack
    if (argc < 2) {
        fprintf(stderr, "usage: %s calibrate|attack\n", argv[0]);
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

    fprintf(stderr, "unknown mode: %s\n", argv[1]);
    return 1;
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

