#!/usr/bin/env python3
# analyze_calib.py
# Reads the output of `./flush_flush calibrate` (lines like "hit 123" / "miss 87")
# and picks a good THRESHOLD value to separate cached (hit) vs evicted (miss) flush timings.
#
# Usage:
#   ./flush_flush calibrate > calib.csv
#   python3 analyze_calib.py calib.csv

import sys          # for reading the filename off the command line and exiting on error
import numpy as np  # for fast array math (mean, percentile, histogram)

def load_calib(path):
    """Parse calib.csv into two lists of cycle counts: hits and misses."""
    hits = []    # will hold cycle counts from lines starting with "hit"
    misses = []  # will hold cycle counts from lines starting with "miss"

    with open(path, "r") as f:        # open the calibration file for reading
        for line in f:                # go through every line in the file
            parts = line.split()      # split "hit 123" into ["hit", "123"]
            if len(parts) != 2:       # skip malformed/blank lines
                continue
            label, value = parts      # label is "hit" or "miss", value is the cycle count as text
            try:
                cycles = int(value)   # convert the cycle count text to an integer
            except ValueError:
                continue               # skip lines where the second field isn't a number

            if label == "hit":        # line was cached before being flushed
                hits.append(cycles)   # record it in the hits list
            elif label == "miss":     # line was already evicted before being flushed
                misses.append(cycles) # record it in the misses list

    return np.array(hits), np.array(misses)  # return both as numpy arrays for easy math


def print_stats(name, data):
    """Print summary statistics for one array of cycle counts."""
    print(f"{name}: n={len(data)}")                                  # how many samples
    print(f"  mean   = {data.mean():.2f}")                           # average cycle count
    print(f"  std    = {data.std():.2f}")                            # spread/noise of the data
    print(f"  min    = {data.min()}")                                # fastest flush observed
    print(f"  max    = {data.max()}")                                # slowest flush observed
    print(f"  p5     = {np.percentile(data, 5):.2f}")                # 5th percentile (lower tail)
    print(f"  p50    = {np.percentile(data, 50):.2f}")               # median
    print(f"  p95    = {np.percentile(data, 95):.2f}")               # 95th percentile (upper tail)


def find_best_threshold(hits, misses):
    """
    Flush+Flush expectation: hit (cached) flushes are SLOWER than miss (evicted)
    flushes, because clflush has real eviction work to do on a cached line.
    So the decision rule is: cycles > THRESHOLD  => treat as "hit" (was cached).
    """
    sorted_hits = np.sort(hits)              # hits sorted ascending, needed for searchsorted
    sorted_misses = np.sort(misses)          # misses sorted ascending, needed for searchsorted

    hit_median = np.median(hits)             # middle value of the hit class
    miss_median = np.median(misses)          # middle value of the miss class
    best_t = (hit_median + miss_median) / 2.0   # the midpoint between the two medians

    # count hits <= best_t (false "miss" calls) and misses > best_t (false "hit" calls),
    # same bookkeeping as before, just evaluated at this ONE threshold instead of a scan
    false_miss = np.searchsorted(sorted_hits, best_t, side="right")
    false_hit = len(sorted_misses) - np.searchsorted(sorted_misses, best_t, side="right")

    best_err = false_miss + false_hit        # total misclassified samples at this threshold
    total = len(hits) + len(misses)          # total number of samples across both sets
    return best_t, best_err, total            # hand back the median-midpoint threshold and its error rate


def main():
    if len(sys.argv) != 2:                       # script expects exactly one argument: the file path
        print("usage: python3 analyze_calib.py <calib.csv>")  # tell the user how to call it
        sys.exit(1)                               # exit with an error code

    path = sys.argv[1]                           # the calibration file path from the command line
    hits, misses = load_calib(path)              # parse the file into hit/miss cycle-count arrays

    if len(hits) == 0 or len(misses) == 0:       # make sure we actually got both kinds of samples
        print("error: need at least one 'hit' and one 'miss' sample in the file")
        sys.exit(1)

    print_stats("hit  (cached line, flush is slower)", hits)    # print stats for the hit population
    print()                                       # blank line for readability
    print_stats("miss (evicted line, flush is faster)", misses)  # print stats for the miss population
    print()

    overlap_lo = max(misses.min(), hits.min())    # start of the region where both sets have samples
    overlap_hi = min(misses.max(), hits.max())    # end of that overlap region
    if overlap_lo <= overlap_hi:                  # if the two distributions actually overlap
        print(f"overlap region: [{overlap_lo}, {overlap_hi}] cycles")  # report how wide the overlap is
    else:
        print("no overlap between hit and miss distributions (clean separation)")  # best case

    best_t, best_err, total = find_best_threshold(hits, misses)  # compute the median-midpoint threshold
    print()
    print(f"suggested THRESHOLD = {best_t:.1f} (midpoint of hit/miss medians)")  # print the recommended threshold
    print(f"misclassified at that threshold: {best_err} / {total} "
          f"({100.0 * best_err / total:.2f}%)")                   # print how good that threshold is


if __name__ == "__main__":   # only run main() when this file is executed directly, not when imported
    main()                    # kick off the script
