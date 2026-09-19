# SPDX-License-Identifier: Apache-2.0
"""Bounded GPU plan construction; producer stages are separate kernel launches."""
import triton
import triton.language as tl


@triton.jit
def count_scan(Lengths, Splits, Active, Counts, Offsets, Chunk, Total, Error,
               BCAP: tl.constexpr, SMAX: tl.constexpr, MAX_CONTEXT: tl.constexpr,
               B: tl.constexpr):
    r = tl.arange(0, B)
    active = tl.load(Active)
    live = (r < active) & (r < BCAP)
    length = tl.load(Lengths + r, live, 0)
    splits = tl.load(Splits + r, live, 1)
    invalid = live & ((length < 0) | (length > MAX_CONTEXT) | (splits < 1) | (splits > SMAX))
    error = (active < 0) | (active > BCAP) | (tl.sum(invalid.to(tl.int32), 0) > 0)
    chunk = tl.cdiv(tl.cdiv(tl.maximum(length, 0), tl.maximum(splits, 1)), 32) * 32
    n = tl.where(live & ~invalid, tl.cdiv(length, tl.maximum(chunk, 1)), 0)
    n = tl.where(error, 0, n)
    inclusive = tl.cumsum(n, 0)
    tl.store(Counts + r, n, r < BCAP)
    tl.store(Chunk + r, chunk, r < BCAP)
    tl.store(Offsets + r, inclusive - n, r < BCAP)
    tl.store(Offsets + BCAP, tl.sum(n, 0))
    tl.store(Total, tl.sum(n, 0))
    tl.store(Error, error.to(tl.int32))


@triton.jit
def emit_records(Lengths, Counts, Offsets, Chunk, R, S, Begin, End, Weight,
                 SMAX: tl.constexpr, TILE: tl.constexpr, SS: tl.constexpr):
    r = tl.program_id(0)
    s = tl.arange(0, SS)
    n = tl.load(Counts + r)
    idx = tl.load(Offsets + r) + s
    chunk = tl.load(Chunk + r)
    begin = s * chunk
    end = tl.minimum(begin + chunk, tl.load(Lengths + r))
    valid = (s < n) & (s < SMAX)
    tl.store(R + idx, r, valid)
    tl.store(S + idx, s, valid)
    tl.store(Begin + idx, begin, valid)
    tl.store(End + idx, end, valid)
    tl.store(Weight + idx, tl.cdiv(end - begin, TILE), valid)


@triton.jit
def make_order_and_packs(Weight, Total, Order, First, Count, NumPacks, Mode,
                         Rules, RULES: tl.constexpr, FORCE: tl.constexpr,
                         G: tl.constexpr, WORKERS: tl.constexpr,
                         BUCKETS: tl.constexpr, E: tl.constexpr, T: tl.constexpr):
    i = tl.arange(0, E)
    total = tl.load(Total)
    valid = i < total
    weight = tl.load(Weight + i, valid, 1)
    max_weight = tl.max(tl.where(valid, weight, 1), 0)
    min_weight = tl.min(tl.where(valid, weight, 2147483647), 0)
    ratio = max_weight * 100 // tl.maximum(min_weight, 1)
    mode = tl.full((), 1, tl.int32)
    for row in range(RULES):
        match = ((total * G >= tl.load(Rules + row * 5)) &
                 (total * G <= tl.load(Rules + row * 5 + 1)) &
                 (ratio >= tl.load(Rules + row * 5 + 2)) &
                 (ratio <= tl.load(Rules + row * 5 + 3)))
        mode = tl.where(match, tl.load(Rules + row * 5 + 4), mode)
    # Uniform batches bypass sorting/packing, except explicit ablation modes.
    mode = tl.where(max_weight == min_weight, 1, mode)
    if FORCE >= 0:
        mode = tl.full((), FORCE, tl.int32)
    tl.store(Mode, mode)
    if (mode == 0) | (mode == 1):
        tl.store(Order + i, i, valid)
        tl.store(NumPacks, 0)
    else:
        # Integer ceil_log2 without floating-point boundary rounding.
        bucket = tl.full((E,), 0, tl.int32)
        for b in tl.static_range(1, BUCKETS):
            bucket += (weight > (1 << (b - 1))).to(tl.int32)
        record_base = tl.full((), 0, tl.int32)
        pack_base = tl.full((), 0, tl.int32)
        p = tl.arange(0, T)
        for reverse in range(BUCKETS):
            b = BUCKETS - 1 - reverse
            member = valid & (bucket == b)
            rank = tl.cumsum(member.to(tl.int32), 0) - 1
            size = tl.sum(member.to(tl.int32), 0)
            tl.store(Order + record_base + rank, i, member)
            tasks = size * G
            width = tl.where(mode == 4, tl.minimum(4, tl.maximum(1, max_weight // (1 << b))), 1)
            packed_tasks = tl.maximum(0, tasks - WORKERS) // width * width
            packed_packs = packed_tasks // width
            pack_count = packed_packs + tasks - packed_tasks
            first = record_base * G + tl.where(p < packed_packs, p * width,
                                                packed_tasks + p - packed_packs)
            tl.store(First + pack_base + p, first, p < pack_count)
            tl.store(Count + pack_base + p, tl.where(p < packed_packs, width, 1), p < pack_count)
            record_base += size
            pack_base += pack_count
        tl.store(NumPacks, pack_base)


@triton.jit
def greedy_lists(Weight, Total, Mode, Heads, Next,
                  G: tl.constexpr, WORKERS: tl.constexpr, W: tl.constexpr,
                  E: tl.constexpr):
    """GPU LPT reference: exact weight order, O(E*T + W*T), bounded small batches.

    The faster heap-based host planner is also supplied; include its upload and
    host planning costs when choosing it as a baseline.
    """
    if tl.load(Mode) == 2:
        worker = tl.arange(0, W)
        record = tl.arange(0, E)
        total = tl.load(Total)
        weights = tl.load(Weight + record, record < total, -1)
        loads = tl.full((W,), 0, tl.int32)
        tails = tl.full((W,), -1, tl.int32)
        tl.store(Heads + worker, -1, worker < WORKERS)
        for _ in range(total):
            largest = tl.max(weights, 0)
            chosen = tl.min(tl.where(weights == largest, record, 2147483647), 0)
            for group in range(G):
                task = chosen * G + group
                minimum = tl.min(tl.where(worker < WORKERS, loads, 2147483647), 0)
                owner = tl.min(tl.where((worker < WORKERS) & (loads == minimum), worker, 2147483647), 0)
                tail = tl.sum(tl.where(worker == owner, tails, 0), 0)
                if tail < 0:
                    tl.store(Heads + owner, task)
                else:
                    tl.store(Next + tail, task)
                tl.store(Next + task, -1)
                loads += tl.where(worker == owner, largest, 0)
                tails = tl.where(worker == owner, task, tails)
            weights = tl.where(record == chosen, -1, weights)
