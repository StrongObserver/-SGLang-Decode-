# SPDX-License-Identifier: Apache-2.0
"""Persistent grouped-query decode with interchangeable task dispatch.

Split geometry follows SGLang; tiled online softmax follows the Triton fused
attention tutorial. Scheduling, record indirection and page/CSR dispatch are
implemented here. See THIRD_PARTY_NOTICES.md for exact source boundaries.
"""
import triton
import triton.language as tl


@triton.jit
def _attention_task(task, mode, Q, K, V, Pages, Indptr, Lengths,
                    R, S, Begin, End, Order, Offsets, Chunk, Active,
                    Partial, Lse, scale,
                    Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
                    KP: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
                    VP: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
                    PT0: tl.constexpr, PT1: tl.constexpr,
                    HQ: tl.constexpr, G: tl.constexpr, GROUP: tl.constexpr,
                    D: tl.constexpr, SMAX: tl.constexpr, PAGE: tl.constexpr,
                    CSR: tl.constexpr, TILE: tl.constexpr, HM: tl.constexpr):
    group = task % G
    record = task // G
    if mode == 0:
        # R0: upper-bound search of request prefix sums on every logical task.
        lo = tl.full((), 0, tl.int32)
        hi = tl.load(Active)
        while lo < hi:
            mid = (lo + hi) // 2
            right = tl.load(Offsets + mid + 1)
            take_right = record >= right
            lo = tl.where(take_right, mid + 1, lo)
            hi = tl.where(take_right, hi, mid)
        request = lo
        split = record - tl.load(Offsets + request)
        chunk = tl.load(Chunk + request)
        start = split * chunk
        end = tl.minimum(start + chunk, tl.load(Lengths + request))
    else:
        # Static LPT lists use original record ids; dynamic queues use permutation P.
        if mode != 2:
            record = tl.load(Order + record)
        request = tl.load(R + record)
        split = tl.load(S + record)
        start = tl.load(Begin + record)
        end = tl.load(End + record)
    heads = group * GROUP + tl.arange(0, HM)
    dims = tl.arange(0, D)
    tokens = tl.arange(0, TILE)
    q = tl.load(Q + request * Q0 + heads[:, None] * Q1 + dims[None, :] * Q2,
                tl.arange(0, HM)[:, None] < GROUP, 0)
    m = tl.full((HM,), -float("inf"), tl.float32)
    denominator = tl.full((HM,), 0, tl.float32)
    accumulator = tl.full((HM, D), 0, tl.float32)
    length = tl.load(Lengths + request)
    for offset in range(start, end, TILE):
        token = offset + tokens
        live = (token < end) & (token < length)
        if CSR:
            base = tl.load(Indptr + request)
            physical = tl.load(Pages + base + token, live, 0)
            inner = tl.full((TILE,), 0, tl.int32)
        else:
            physical = tl.load(Pages + request * PT0 + (token // PAGE) * PT1, live, 0)
            inner = token % PAGE
        # Use 64-bit addresses even when metadata indices are int32.
        kaddr = physical.to(tl.int64) * KP + inner * KT + group * KH
        vaddr = physical.to(tl.int64) * VP + inner * VT + group * VH
        key = tl.load(K + kaddr[None, :] + dims[:, None] * KD, live[None, :], 0)
        value = tl.load(V + vaddr[:, None] + dims[None, :] * VD, live[:, None], 0)
        score = tl.dot(q, key).to(tl.float32) * scale
        score = tl.where(live[None, :], score, -float("inf"))
        new_m = tl.maximum(m, tl.max(score, 1))
        alpha = tl.exp(m - new_m)
        probability = tl.exp(score - new_m[:, None])
        accumulator = accumulator * alpha[:, None]
        accumulator += tl.dot(probability.to(value.dtype), value)
        denominator = denominator * alpha + tl.sum(probability, 1)
        m = new_m
    slot = (request * HQ + heads) * SMAX + split
    tl.store(Partial + slot[:, None] * D + dims[None, :],
             accumulator / denominator[:, None], tl.arange(0, HM)[:, None] < GROUP)
    tl.store(Lse + slot, m + tl.log(denominator), tl.arange(0, HM) < GROUP)


@triton.jit
def persistent_decode(Q, K, V, Pages, Indptr, Lengths, R, S, Begin, End, Order,
                       Offsets, Chunk, Active, Total, Mode, First, Count, NumPacks,
                       Counter, StaticHead, StaticNext, Partial, Lse, scale,
                       Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
                       KP: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
                       VP: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
                       PT0: tl.constexpr, PT1: tl.constexpr,
                       HQ: tl.constexpr, G: tl.constexpr, GROUP: tl.constexpr,
                       D: tl.constexpr, SMAX: tl.constexpr, PAGE: tl.constexpr,
                       CSR: tl.constexpr, TILE: tl.constexpr, HM: tl.constexpr,
                       WORKERS: tl.constexpr):
    worker = tl.program_id(0)
    mode = tl.load(Mode)
    total = tl.load(Total) * G
    dynamic = mode >= 3
    if dynamic:
        # A scalar Triton atomic is one logical claim, broadcast through SSA.
        cursor = tl.atomic_add(Counter, 1, sem="relaxed")
        limit = tl.load(NumPacks)
    elif mode == 2:
        cursor = tl.load(StaticHead + worker)
        limit = total
    else:
        cursor = worker
        limit = total
    while (cursor >= 0) & (cursor < limit):
        if dynamic:
            first = tl.load(First + cursor)
            count = tl.load(Count + cursor)
        else:
            first = cursor
            count = 1
        for item in range(count):
            _attention_task(first + item, mode, Q, K, V, Pages, Indptr, Lengths,
                            R, S, Begin, End, Order, Offsets, Chunk, Active, Partial, Lse,
                            scale, Q0, Q1, Q2, KP, KT, KH, KD, VP, VT, VH, VD, PT0, PT1,
                            HQ, G, GROUP, D, SMAX, PAGE, CSR, TILE, HM)
        if dynamic:
            cursor = tl.atomic_add(Counter, 1, sem="relaxed")
        elif mode == 2:
            cursor = tl.load(StaticNext + cursor)
        else:
            cursor += WORKERS
