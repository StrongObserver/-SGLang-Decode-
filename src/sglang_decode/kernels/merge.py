# SPDX-License-Identifier: Apache-2.0
"""Merge normalized partial outputs with natural-log LSE, in original split order."""
import triton
import triton.language as tl


@triton.jit
def merge_splits(Partial, Lse, Counts, Active, Error, Out,
                 HQ: tl.constexpr, D: tl.constexpr, SMAX: tl.constexpr):
    request, head = tl.program_id(0), tl.program_id(1)
    dims = tl.arange(0, D)
    n = tl.load(Counts + request)
    m = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    accumulator = tl.full((D,), 0, tl.float32)
    # Never load stale slots >= n, including stale NaNs from a larger previous batch.
    for split in range(n):
        slot = (request * HQ + head) * SMAX + split
        lse = tl.load(Lse + slot)
        partial = tl.load(Partial + slot * D + dims)
        new_m = tl.maximum(m, lse)
        old_weight = tl.exp(m - new_m)
        weight = tl.exp(lse - new_m)
        accumulator = accumulator * old_weight + partial * weight
        denominator = denominator * old_weight + weight
        m = new_m
    result = accumulator / tl.maximum(denominator, 1.0e-30)
    # Standalone empty rows are defined as zero; invalid metadata is conspicuous.
    result = tl.where(tl.load(Error) != 0, float("nan"), result)
    tl.store(Out + (request * HQ + head) * D + dims, result)
