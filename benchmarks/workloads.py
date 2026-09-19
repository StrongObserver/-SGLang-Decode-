"""Preserved A/B geometries and separate synthetic holdout inputs."""
import math
import torch

WORKLOADS = {
    "A": [512] * 24 + [8192] * 8,
    "B": [256] * 24 + [1536] * 4 + [16384] * 4,
    "B64": ([256] * 24 + [1536] * 4 + [16384] * 4) * 2,
    "uniform": [1024] * 32,
    "holdout": [33, 255, 513, 1025, 3073, 8191, 16383],
}


def split_hints(lengths, maximum=16):
    """Explicit fixture rule yielding the documented A/B boundaries.

    Real integration consumes SGLang num_kv_splits directly; this is not a claim
    that every upstream heuristic/device chooses these exact fixture counts.
    """
    chunk = max(1, math.ceil(max(lengths, default=0) / maximum))
    return [max(1, min(maximum, math.ceil(n / chunk))) for n in lengths]


def make_inputs(config, lengths, seed=17, device="cuda", dtype=torch.bfloat16):
    generator = torch.Generator(device=device).manual_seed(seed)
    pages_per_request = [math.ceil(n / config.page_size) for n in lengths]
    physical_pages = max(1, sum(pages_per_request))
    permutation = torch.randperm(physical_pages, generator=generator, device=device)
    table = torch.full((config.batch_capacity, config.max_pages), -1,
                       dtype=torch.int32, device=device)
    cursor = 0
    for row, count in enumerate(pages_per_request):
        table[row, :count] = permutation[cursor:cursor + count].int()
        cursor += count
    shape = (physical_pages, config.page_size, config.kv_heads, config.head_dim)
    k = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    v = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    q = torch.randn((config.batch_capacity, config.query_heads, config.head_dim),
                    generator=generator, device=device, dtype=dtype)
    return q, k, v, table


def update_plan(engine, lengths):
    p = engine.plan
    p.lengths.zero_()
    p.splits.fill_(1)
    p.active.fill_(len(lengths))
    p.lengths[:len(lengths)].copy_(torch.tensor(lengths, device=p.device, dtype=torch.int32))
    p.splits[:len(lengths)].copy_(torch.tensor(split_hints(lengths, engine.config.max_splits),
                                            device=p.device, dtype=torch.int32))
