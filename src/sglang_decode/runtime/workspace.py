"""Long-lived, instance-private device buffers. Allocate before graph capture."""
from dataclasses import dataclass
import torch
from ..config import DecodeConfig


class PlanBuffers:
    def __init__(self, config: DecodeConfig, device, rules=()):
        self.config = config
        self.device = torch.device(device)
        def ints(n):
            return torch.zeros(n, dtype=torch.int32, device=device)
        b, e, t = config.batch_capacity, config.record_capacity, config.task_capacity
        self.lengths, self.splits = ints(b), ints(b)
        self.active = ints(1)
        self.counts, self.offsets, self.chunk = ints(b), ints(b + 1), ints(b)
        self.request, self.split, self.begin, self.end, self.weight = [ints(e) for _ in range(5)]
        self.order, self.pack_first, self.pack_count = ints(e), ints(t), ints(t)
        self.record_count, self.num_packs, self.mode, self.error = [ints(1) for _ in range(4)]
        self.static_head, self.static_next = ints(config.workers), ints(t)
        rows = [[r.min_tasks, r.max_tasks, r.min_ratio, r.max_ratio, r.mode] for r in rules]
        self.rules = torch.tensor(rows or [[0, 0, 100, 100, 1]], dtype=torch.int32, device=device)
        self.rule_count = len(rows)


@dataclass
class LayerBuffers:
    counter: torch.Tensor
    partial: torch.Tensor
    lse: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, config, device, dtype=torch.bfloat16):
        shape = (config.batch_capacity, config.query_heads, config.max_splits)
        return cls(torch.zeros(1, dtype=torch.int32, device=device),
                   torch.empty((*shape, config.head_dim), dtype=torch.float32, device=device),
                   torch.empty(shape, dtype=torch.float32, device=device),
                   torch.empty((config.batch_capacity, config.query_heads, config.head_dim),
                               dtype=dtype, device=device))
