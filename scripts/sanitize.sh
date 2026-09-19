#!/usr/bin/env bash
# Optional future diagnostics; NEVER called by static_check.py.
set -euo pipefail
if [[ "${1:-}" != "--run-gpu" ]]; then
  echo 'Explicit --run-gpu required; this command compiles and runs kernels.' >&2
  exit 2
fi
export DECODE_RUN_GPU_TESTS=1
compute-sanitizer --tool memcheck --target-processes all python3 -m pytest -o addopts='' -m gpu tests/gpu
compute-sanitizer --tool synccheck --target-processes all python3 -m pytest -o addopts='' -m gpu tests/gpu
