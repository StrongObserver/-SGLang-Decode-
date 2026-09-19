"""Capture-time capacities and immutable attention geometry."""
from dataclasses import dataclass
from enum import IntEnum


class Mode(IntEnum):
    R0 = 0  # compact prefix-search round robin
    R1 = 1  # direct-record round robin
    STATIC = 2  # longest-processing-time greedy lists
    D1 = 3  # one logical task per atomic claim
    D4 = 4  # bounded short-task packs


@dataclass(frozen=True)
class DecodeConfig:
    batch_capacity: int = 64
    max_splits: int = 16
    max_context: int = 32768
    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    page_size: int = 16
    workers: int = 108
    block_kv: int = 128
    split_alignment: int = 32

    def __post_init__(self):
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.query_heads % self.kv_heads or self.group_size not in (1, 2, 4, 8, 16):
            raise ValueError("supported GQA ratios: 1, 2, 4, 8, 16")
        if self.head_dim not in (64, 128, 256):
            raise ValueError("supported head dimensions: 64, 128, 256")
        if self.block_kv != 128 or self.split_alignment != 32:
            raise ValueError("this kernel specialization uses tile=128 and split alignment=32")
        if self.record_capacity > 4096 or self.task_capacity >= 2**31:
            raise ValueError("planner bound exceeded; choose another instance or upstream")

    @property
    def group_size(self):
        return self.query_heads // self.kv_heads

    @property
    def record_capacity(self):
        return self.batch_capacity * self.max_splits

    @property
    def task_capacity(self):
        return self.record_capacity * self.kv_heads

    @property
    def max_pages(self):
        return (self.max_context + self.page_size - 1) // self.page_size

    @property
    def buckets(self):
        return ((self.max_context + self.block_kv - 1) // self.block_kv - 1).bit_length() + 1
