#!/bin/bash
# run_dmp.sh - build dmp_test, record machine setup, run independent trials, analyze.
# Usage: bash run_dmp.sh [runs=5] [trials=1000] [cpu=2]

RUNS=${1:-5}                                   # number of independent process runs
TRIALS=${2:-1000}                              # measured trials per condition per run
CPU=${3:-2}                                    # core to pin to
OUT=results                                    # output directory

mkdir -p "$OUT"                                # create the results directory
gcc -O1 -o dmp_test dmp_test.c || exit 1       # build the measurement program

{                                              # collect setup info into one file
  echo "date: $(date -Is)"                     # when the experiment ran
  echo "host: $(hostname)"                     # which lab machine
  echo "kernel: $(uname -r)"                   # kernel version
  echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"  # counter permission level
  echo "pinned_cpu: $CPU"                      # which core this run was pinned to
  echo "runs: $RUNS  trials: $TRIALS"          # other run params, for the record
  lscpu                                        # CPU model, caches, flags
} > "$OUT/setup.txt" 2>&1                      # save everything (stderr included)

for r in $(seq 1 "$RUNS"); do                  # one separate process per run for independence
  ./dmp_test "$r" "$TRIALS" "$CPU" > "$OUT/raw_run$r.csv" || exit 1  # raw data for this run
done                                           # end of run loop

head -n 1 "$OUT/raw_run1.csv" > "$OUT/raw_all.csv"   # keep a single header line
for r in $(seq 1 "$RUNS"); do                  # append every run's data rows
  tail -n +2 "$OUT/raw_run$r.csv" >> "$OUT/raw_all.csv"  # skip each run's header (only run 1 has one)
done                                           # end of merge loop

python3 analyze_dmp.py "$OUT/raw_all.csv" "$OUT" | tee "$OUT/summary.txt"  # stats, plots, verdict
