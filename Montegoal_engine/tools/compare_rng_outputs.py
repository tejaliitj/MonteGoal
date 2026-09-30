#!/usr/bin/env python3
"""Check that the RNGs give statistically consistent probabilities.

Reads results/<rng>.txt written by run_compare_rngs.bat (the table printed by
main.cu: Team, QF %, SF %, Final %, Champion %) and, for every pair of RNGs and
every stage, reports the largest gap and the largest z-score:

    z = (p1 - p2) / sqrt(pbar * (1 - pbar) * (1/n1 + 1/n2))

The RNGs use different random numbers, so exact values differ; only the
probabilities should agree. With 48 teams x 4 stages, |z| up to ~4 is normal
noise. Consistently larger values would mean a real difference (or a bug).
Rounding of the printed values is included in the variance.
Speed can only be compared fairly if this check passes.
Standard library only.
"""
import argparse, glob, itertools, math, os, re, sys

ROW = re.compile(r"^(.+?)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s*$")
HEAD = re.compile(r"MonteGoal:\s+(\d+)\s+trials.*RNG\s+(\S+),")
STAGES = ["QF", "SF", "Final", "Champion"]
# main.cu prints percentages with 2 decimals => each value carries rounding error
# of variance (0.0001)^2 / 12 in probability units; include it so big runs are not
# flagged just because of rounding.
ROUND_VAR = (1e-4) ** 2 / 12


def parse(path):
    n, rng, rows = None, None, {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = HEAD.search(line)
            if m:
                n, rng = int(m.group(1)), m.group(2)
                continue
            m = ROW.match(line.strip())
            if m:
                rows[m.group(1).strip()] = [float(m.group(i)) / 100.0 for i in range(2, 6)]
    if n is None or not rows:
        return None
    return rng, n, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    runs = {}
    for p in sorted(glob.glob(os.path.join(args.results_dir, "*.txt"))):
        r = parse(p)
        if r:
            runs[r[0]] = r[1:]
    if len(runs) < 2:
        sys.exit("need at least two RNG result files in " + args.results_dir)

    for a, b in itertools.combinations(sorted(runs), 2):
        (na, ra), (nb, rb) = runs[a], runs[b]
        print(f"{a} vs {b}  ({na:,} vs {nb:,} trials)")
        worst_overall = 0.0
        for s, name in enumerate(STAGES):
            max_gap, max_z, who = 0.0, 0.0, ""
            for team in ra:
                if team not in rb:
                    continue
                pa, pb = ra[team][s], rb[team][s]
                pbar = (pa * na + pb * nb) / (na + nb)
                var = pbar * (1 - pbar) * (1 / na + 1 / nb)
                var += 2 * ROUND_VAR   # printed values are rounded to 0.01 pp
                z = abs(pa - pb) / math.sqrt(var) if var > 0 else 0.0
                max_gap = max(max_gap, abs(pa - pb))
                if z > max_z:
                    max_z, who = z, team
            worst_overall = max(worst_overall, max_z)
            print(f"  {name:<9} max |dp| = {100 * max_gap:.4f} pp   max |z| = {max_z:.2f} ({who})")
        print("  ->", "consistent" if worst_overall < 4.5 else "CHECK: larger than expected noise", "\n")


if __name__ == "__main__":
    main()
