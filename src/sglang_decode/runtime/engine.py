"""Plan/execute API shared by the standalone harness and the SGLang adapter."""
from dataclasses import dataclass
import torch
from ..config import DecodeConfig
from ..planning.builder import build_plan
from ..kernels.attention import persistent_decode
from ..kernels.merge import merge_splits
from .workspace import PlanBuffers, LayerBuffers


@dataclass(frozen=True)
class PlanKey:
    """Caller-supplied sequence identity; pointer equality alone is insufficient."""
    sequence_layout: str
    split_layout: str
    head_layout: tuple
    attention_window: str = "full"


class DecodeEngine:
    def __init__(self, config: DecodeConfig, device="cuda", dtype=torch.bfloat16,
                 layers=1, rules=(), force_mode=-1):
        self.config, self.force_mode = config, force_mode
        self.plan = PlanBuffers(config, device, rules)
        self.plan.has_static_rule = any(r.mode == 2 for r in rules)
        self.layers = [LayerBuffers.allocate(config, device, dtype) for _ in range(layers)]

    def prepare(self):
        """Call once per compatible group per step, including on every graph replay."""
        build_plan(self.plan, self.force_mode)

    def execute(self, q, k, v, pages, layer=0, *, indptr=None, scale=None):
        """KV layout: [pages,page_size,kv_heads,D], or CSR [tokens,kv_heads,D].

        Metadata and Q/KV must be ready on the current CUDA stream. Fixed buffers
        are reused; consumers must finish before reusing this execution instance.
        """
        c, p, out = self.config, self.plan, self.layers[layer]
        if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("BF16/FP16 Q/K/V of identical dtype are required")
        if q.shape != (c.batch_capacity, c.query_heads, c.head_dim):
            raise ValueError("Q must use the capture capacity; pad inactive rows")
        if indptr is None:
            if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
                raise ValueError("paged KV must have matching [page,token,head,dim] shapes")
            if k.shape[1:] != (c.page_size, c.kv_heads, c.head_dim):
                raise ValueError("KV geometry differs from the capture configuration")
            if pages.shape != (c.batch_capacity, c.max_pages):
                raise ValueError("page table capacity mismatch")
            ks, vs = k.stride(), v.stride()
            pt0, pt1 = pages.stride()
            indptr_arg = p.offsets  # unused by the paged specialization
            csr = False
        else:
            if k.ndim != 3 or v.shape != k.shape or k.shape[1:] != (c.kv_heads, c.head_dim):
                raise ValueError("CSR KV must be [token,head,dim]")
            if c.page_size != 1 or pages.ndim != 1:
                raise ValueError("CSR integration uses one-token physical pages")
            ks = (k.stride(0), 0, k.stride(1), k.stride(2))
            vs = (v.stride(0), 0, v.stride(1), v.stride(2))
            pt0, pt1, indptr_arg, csr = 0, 0, indptr, True
        out.counter.zero_()  # captured operation: every layer, every replay
        persistent_decode[(c.workers,)](
            q, k, v, pages, indptr_arg, p.lengths, p.request, p.split, p.begin, p.end,
            p.order, p.offsets, p.chunk, p.active, p.record_count, p.mode,
            p.pack_first, p.pack_count, p.num_packs, out.counter, p.static_head,
            p.static_next, out.partial, out.lse, c.head_dim ** -0.5 if scale is None else scale,
            *q.stride(), *ks, *vs, pt0, pt1, c.query_heads, c.kv_heads, c.group_size,
            c.head_dim, c.max_splits, c.page_size, csr, c.block_kv, max(16, c.group_size),
            c.workers, num_warps=4)
        merge_splits[(c.batch_capacity, c.query_heads)](
            out.partial, out.lse, p.counts, p.active, p.error, out.output,
            c.query_heads, c.head_dim, c.max_splits, num_warps=4)
        return out.output
