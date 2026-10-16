# Prime + Probe — Frozen Thresholds, Ways/Stride per Machine

Threshold is the attacker/reference probe-time ratio (permille); detection fires
when the ratio exceeds it. ways/stride match each machine's real L1d geometry.

| Machine | ways | stride | threshold (atk/ref) |
|---|---:|---:|---:|
| Crux (i7-9700) | 8 | 4096 | 1.259 |
| Upgrade (i7-8700) | 8 | 4096 | 1.263 |
| Charnwood (i7-6700) | 8 | 4096 | 1.244 |
| Ookay (i7-7700) | 8 | 4096 | 1.304 |
| Sunbird (Xeon E5-2680 v3) | 8 | 4096 | 1.119 |
| Artemisia (Xeon Gold 5420+) | 12 | 4096 | 0.931 |
| Skylark (EPYC 7532) | 8 | 4096 | 1.291 |
| Thunderbird (Neoverse N1) | 4 | 16384 | 1.499 |

# Build
```cd prime_probe``` \
```gcc -O0 -o primeprobe primeprobe.c```                  (x86) \
```gcc -O0 -march=armv8-a -o primeprobe primeprobe.c```   (ARM)

# Run (calibrates, then runs the attack; trace + summary to stdout)
```taskset -c <core> ./primeprobe <ways> <stride> > <machine>/<machine>_results.csv``` \

The run self-calibrates the threshold each time (printed in the CSV header), so
no separate calibrate step is needed. `<machine>/<machine>_results.csv` holds the
calibration header (hit/miss p5/p50/p95 ratios + chosen threshold), a sampled
per-round trace (round, secret, ratio_milli, detected), and a final summary line
with accuracy / FP rate / FN rate and the tp/tn/fp/fn confusion counts.

Timer: self-referenced ratio probe(attacker)/probe(reference) over `rdtscp`
(x86) or `cntvct_el0` (ARM); the ratio cancels DVFS so no absolute-cycle
threshold is needed. Set `PRIMEPROBE_PERF=1` to use perf cpu-cycles instead, or
`PRIMEPROBE_DUMP=1` to dump raw hit/miss samples for a histogram.
