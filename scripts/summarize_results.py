#!/usr/bin/env python3
"""Summarize actual CSV samples without imputing historical measurements."""
import argparse
import csv
from collections import defaultdict
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples")
    args = parser.parse_args()
    groups = defaultdict(list)
    with open(args.samples, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                raise SystemExit("failed rounds present; review them before publishing a summary")
            groups[row["variant"]].append(float(row.get("wall_us", row["gpu_us"])))
    for name, values in groups.items():
        print(f"{name}: median={statistics.median(values):.3f} us "
              f"range=[{min(values):.3f}, {max(values):.3f}] n={len(values)} round means")
    if {"R1", "STATIC_CPU", "D1", "D4"} <= groups.keys():
        static = min(("R1", "STATIC_CPU"), key=lambda n: statistics.median(groups[n]))
        baseline = statistics.median(groups[static])
        for name in ("D1", "D4"):
            delta = 1 - statistics.median(groups[name]) / baseline
            separated = max(groups[name]) < min(groups[static])
            print(f"{name} vs {static}: {delta:.2%}; >=3% and disjoint ranges: "
                  f"{delta >= .03 and separated}; independent holdout still required")


if __name__ == "__main__":
    main()
