"""Slow FP32 numerical oracle; never used by the optimized dispatch path."""
import torch


def paged_attention(q, k, v, pages, lengths, page_size, scale=None):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    group = q.shape[1] // k.shape[2]
    out = torch.zeros_like(q, dtype=torch.float32)
    for r, length in enumerate(lengths):
        if length == 0:
            continue
        token = torch.arange(length, device=q.device)
        physical = pages[r, token // page_size].long()
        key = k[physical, token % page_size].float().repeat_interleave(group, dim=1)
        value = v[physical, token % page_size].float().repeat_interleave(group, dim=1)
        score = torch.einsum("hd,thd->ht", q[r].float(), key) * scale
        out[r] = torch.einsum("ht,thd->hd", score.softmax(-1), value)
    return out
