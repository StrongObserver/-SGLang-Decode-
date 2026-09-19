#!/usr/bin/env python3
"""Freeze one bounded policy rule from independent measured CSVs; never run kernels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

MODES = {"R1": 1, "STATIC_CPU": 2, "D1": 3, "D4": 4}


def read(path):
    groups = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                raise ValueError("failed samples need investigation; do not silently drop them")
            groups.setdefault(row["variant"], []).append(float(row["wall_us"]))
    for name in MODES:
        if len(groups.get(name, [])) < 5:
            raise ValueError(f"at least five full-wall rounds required for {name}")
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration", type=Path)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs=2, type=int, required=True)
    parser.add_argument("--ratio", nargs=2, type=int, required=True)
    args = parser.parse_args()
    if args.calibration.resolve() == args.holdout.resolve():
        parser.error("independent holdout artifact required")
    calibration, holdout = read(args.calibration), read(args.holdout)
    median = lambda group, name: statistics.median(group[name])
    static = min(("R1", "STATIC_CPU"), key=lambda n: median(calibration, n))
    candidate = min(MODES, key=lambda n: median(calibration, n))
    if candidate in ("D1", "D4"):
        improvement = 1 - median(calibration, candidate) / median(calibration, static)
        if improvement < .03 or max(calibration[candidate]) >= min(calibration[static]):
            candidate = static
    holdout_static = min(("R1", "STATIC_CPU"), key=lambda n: median(holdout, n))
    if median(holdout, candidate) > 1.02 * median(holdout, holdout_static):
        raise SystemExit("holdout regression exceeds 2%; no policy published")
    if candidate in ("D1", "D4") and (
            median(holdout, candidate) > .97 * median(holdout, holdout_static)
            or max(holdout[candidate]) >= min(holdout[holdout_static])):
        raise SystemExit("dynamic holdout benefit is below the adoption gate")
    if candidate == "STATIC_CPU":
        raise SystemExit("host static wins: retain it as baseline; do not substitute GPU LPT in AUTO")
    evidence = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in (args.calibration, args.holdout)]
    artifact = dict(status="validated", evidence=evidence,
                    rules=[dict(min_tasks=args.tasks[0], max_tasks=args.tasks[1],
                                min_ratio=args.ratio[0], max_ratio=args.ratio[1], mode=MODES[candidate])])
    with args.output.open("x") as f:
        json.dump(artifact, f, indent=2)


if __name__ == "__main__":
    main()
