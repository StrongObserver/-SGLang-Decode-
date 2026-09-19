#!/usr/bin/env python3
"""Apply/revert one explicit import change in an exact pinned SGLang checkout."""
import argparse
from pathlib import Path
import subprocess

PIN = "7e257cd666c0d639626487987ea8e590da1e9395"
OLD = "from sglang.srt.layers.attention.triton_backend import TritonAttnBackend"
NEW = "from sglang_decode.integrations.sglang_backend import ScheduledTritonBackend as TritonAttnBackend"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
    if sha != PIN:
        raise SystemExit(f"expected {PIN}, got {sha}; do not patch a drifting API")
    path = args.checkout / "python/sglang/srt/model_executor/model_runner.py"
    before = path.read_text()
    source, target = (NEW, OLD) if args.revert else (OLD, NEW)
    if before.count(source) != 1:
        raise SystemExit("expected exactly one import site; no file changed")
    path.write_text(before.replace(source, target))
    print(f"updated {path}")


if __name__ == "__main__":
    main()
