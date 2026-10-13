#!/usr/bin/env python3
# analyze_dmp.py
# Reads raw CSV from dmp_test (run,trial,cond,j,cycles,l1_miss,llc_miss), prints a summary
# table, statistical tests, a pre-declared verdict, and saves plots.
#
# Usage:
#   python3 analyze_dmp.py results/raw_all.csv results

import sys                                   # command line arguments and exit
import numpy as np                           # array math
import matplotlib                            # plotting backend selection
matplotlib.use("Agg")                        # render to files, no display needed on lab machines
import matplotlib.pyplot as plt              # plotting API
from scipy.stats import mannwhitneyu         # non-parametric two-sample test

NAMES = ["TEST", "C1_nonptr", "C2_notrain", "C3_neighbor", "C4_seq", "C5_deref"]  # cond id -> name
TEST, NONPTR, NOTRAIN, NEIGHBOR, SEQ, DEREF = range(6)  # readable cond ids matching dmp_test.c
M = 32                                       # probed lines per trial (must match dmp_test.c)
MARGIN = 0.10                                # TEST fast-fraction must beat every control by this much
BOOT = 2000                                  # bootstrap resamples for confidence intervals
rng = np.random.default_rng(0)               # fixed seed so the analysis is reproducible


def load(path):
    """Load the CSV into a dict of int64 columns."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)  # all numeric columns
    cols = ["run", "trial", "cond", "j", "cycles", "l1", "llc"]         # column names in file order
    return {c: raw[:, i] for i, c in enumerate(cols)}                   # name -> column array


def per_trial(d, cond, thr):
    """Return per-trial arrays for one condition: run id, fast fraction, l1, llc, median cycles."""
    mask = d["cond"] == cond                                  # rows belonging to this condition
    cyc = d["cycles"][mask].reshape(-1, M)                    # one row per trial, one column per probe
    runs = d["run"][mask][::M]                                # run id of each trial
    l1 = d["l1"][mask][::M]                                   # L1-miss counter delta of each trial
    llc = d["llc"][mask][::M]                                 # LLC-miss counter delta of each trial
    fast = (cyc < thr).mean(axis=1)                           # fraction of probes faster than threshold
    return runs, fast, l1, llc, np.median(cyc, axis=1)        # bundle everything per trial


def boot_ci(x):
    """95% bootstrap confidence interval of the mean of x."""
    means = [rng.choice(x, size=len(x)).mean() for _ in range(BOOT)]  # resampled means
    return np.percentile(means, [2.5, 97.5])                  # lower and upper bound


def main():
    if len(sys.argv) != 3:                                    # need csv path and output dir
        print("usage: python3 analyze_dmp.py <raw.csv> <outdir>")  # explain usage
        sys.exit(1)                                           # error exit
    path, outdir = sys.argv[1], sys.argv[2]                   # input file and output directory
    d = load(path)                                            # parse the data

    hit_med = np.median(d["cycles"][d["cond"] == DEREF])      # cached latency (upper bound control)
    cold_med = np.median(d["cycles"][d["cond"] == NOTRAIN])   # uncached latency (cold baseline)
    thr = (hit_med + cold_med) / 2.0                          # midpoint: below = "prefetched/cached"
    print(f"cached median (C5) = {hit_med:.0f} cyc, cold median (C2) = {cold_med:.0f} cyc, "
          f"threshold = {thr:.1f} cyc")                       # report calibration
    if cold_med - hit_med < 20:                               # timing cannot separate hit from miss
        print("WARNING: cached/cold latencies too close; timing is unreliable on this machine")

    runs_all = sorted(set(d["run"]))                          # list of run ids present
    res = {}                                                  # cond -> per-trial tuple
    print(f"\n{'condition':<13}{'fast_frac':>10}{'95% CI':>20}{'med_cyc':>9}{'med_L1miss':>12}{'med_LLCmiss':>13}")
    for c in range(6):                                        # one summary line per condition
        res[c] = per_trial(d, c, thr)                         # per-trial arrays
        _, fast, l1, llc, _ = res[c]                          # unpack what we print
        lo, hi = boot_ci(fast)                                # confidence interval on fast fraction
        med_cyc = np.median(d["cycles"][d["cond"] == c])      # median probe latency
        print(f"{NAMES[c]:<13}{fast.mean():>10.3f}{f'[{lo:.3f},{hi:.3f}]':>20}"
              f"{med_cyc:>9.0f}{np.median(l1):>12.0f}{np.median(llc):>13.0f}")

    print("\nper-run fast fraction (rows = condition, cols = run):")  # reliability across processes
    print(f"{'':<13}" + "".join(f"{'run'+str(r):>8}" for r in runs_all))  # header row
    per_run = {}                                              # cond -> list of per-run fast fractions
    for c in range(6):                                        # one row per condition
        runs, fast, _, _, _ = res[c]                          # per-trial data
        per_run[c] = [fast[runs == r].mean() for r in runs_all]  # mean within each run
        print(f"{NAMES[c]:<13}" + "".join(f"{v:>8.3f}" for v in per_run[c]))  # print the row

    print("\nTEST vs controls (Mann-Whitney U on per-trial fast fraction, Cohen's d):")  # significance
    for c in (NONPTR, NOTRAIN, NEIGHBOR, SEQ):                # compare TEST against each control
        a, b = res[TEST][1], res[c][1]                        # per-trial fast fractions
        p = mannwhitneyu(a, b, alternative="greater").pvalue  # one-sided: TEST > control
        sd = np.sqrt((a.var() + b.var()) / 2.0) or 1e-9       # pooled std (avoid divide by zero)
        print(f"  TEST vs {NAMES[c]:<12} p={p:.3g}  d={(a.mean() - b.mean()) / sd:.2f}")

    # Pre-declared verdict: TEST must beat C1, C2, C3 in EVERY run by MARGIN, and show fewer L1 misses than C2.
    ok_runs = all(per_run[TEST][i] > max(per_run[k][i] for k in (NONPTR, NOTRAIN, NEIGHBOR)) + MARGIN
                  for i in range(len(runs_all)))              # per-run consistency check
    ok_ctr = np.median(res[TEST][2]) < np.median(res[NOTRAIN][2])  # counter agrees with timing
    have_ctr = np.median(res[NOTRAIN][2]) > 0                 # counters actually worked
    print("\nVERDICT:")                                       # final decision
    if ok_runs and (ok_ctr or not have_ctr):                  # timing (and counters if present) agree
        print("  SUPPORTS DMP presence" + ("" if have_ctr else " (timing only; counters unavailable)"))
    else:
        print("  INSUFFICIENT / no DMP detected: TEST not reliably above controls in every run"
              + ("" if ok_ctr or not have_ctr else ", and L1-miss counter disagrees"))

    fig, ax = plt.subplots(figsize=(7, 4))                    # histogram of probe latencies
    for c in range(6):                                        # one curve per condition
        ax.hist(d["cycles"][d["cond"] == c], bins=np.arange(0, 600, 10), histtype="step",
                label=NAMES[c], density=True)                 # normalized so conditions are comparable
    ax.axvline(thr, color="k", ls="--", lw=0.8)               # threshold line
    ax.set_xlabel("probe latency (cycles)")                   # x label
    ax.set_ylabel("density")                                  # y label
    ax.set_yscale("log")                                      # tails matter, use log scale
    ax.legend(fontsize=7)                                     # identify the curves
    fig.tight_layout()                                        # avoid clipped labels
    fig.savefig(f"{outdir}/latency_hist.png", dpi=150)        # save plot 1

    fig, ax = plt.subplots(figsize=(7, 4))                    # per-run fast-fraction bars (reliability)
    w = 0.8 / len(runs_all)                                   # bar width per run
    for i, r in enumerate(runs_all):                          # one bar group per run
        ax.bar(np.arange(6) + i * w, [per_run[c][i] for c in range(6)], w, label=f"run {r}")  # bars
    ax.set_xticks(np.arange(6) + 0.4 - w / 2)                 # center tick labels under groups
    ax.set_xticklabels(NAMES, rotation=20, fontsize=8)        # condition names
    ax.set_ylabel(f"fraction of probes < {thr:.0f} cyc")      # y label with threshold
    ax.legend(fontsize=7)                                     # run legend
    fig.tight_layout()                                        # avoid clipped labels
    fig.savefig(f"{outdir}/fast_fraction_per_run.png", dpi=150)  # save plot 2

    fig, ax = plt.subplots(figsize=(7, 4))                    # perf-counter box plot
    ax.boxplot([res[c][2] for c in range(6)], labels=NAMES, showfliers=False)  # L1 misses per trial
    ax.set_ylabel("L1 load misses in probe window")           # y label
    plt.setp(ax.get_xticklabels(), rotation=20, fontsize=8)   # readable tick labels
    fig.tight_layout()                                        # avoid clipped labels
    fig.savefig(f"{outdir}/l1_miss_box.png", dpi=150)         # save plot 3


if __name__ == "__main__":                                    # run only when executed directly
    main()                                                    # start analysis
