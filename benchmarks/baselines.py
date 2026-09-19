"""Optional external baselines using identical Q/KV and effective token ranges."""
import importlib.util
from pathlib import Path
import torch
from .workloads import split_hints


def upstream_call(config, inputs, lengths):
    path = Path(__file__).resolve().parents[1] / "third_party/sglang/decode_attention.py"
    spec = importlib.util.spec_from_file_location("pinned_sglang_decode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q, k, v, pages = inputs
    b = len(lengths)
    pointers, indices = [0], []
    for r, length in enumerate(lengths):
        token = torch.arange(length, device=q.device)
        indices.append((pages[r, token // config.page_size].long() * config.page_size
                        + token % config.page_size).int())
        pointers.append(pointers[-1] + length)
    indptr = torch.tensor(pointers, device=q.device, dtype=torch.int32)
    indices = torch.cat(indices)
    splits = torch.tensor(split_hints(lengths, config.max_splits), device=q.device, dtype=torch.int32)
    partial = torch.empty((b, config.query_heads, config.max_splits, config.head_dim),
                          device=q.device, dtype=torch.float32)
    lse = torch.empty(partial.shape[:-1], device=q.device, dtype=torch.float32)
    out = torch.empty_like(q[:b])
    key = k.view(-1, config.kv_heads, config.head_dim)
    value = v.view_as(key)
    def call():
        module.decode_attention_fwd(q[:b], key, value, out, indptr, indices, partial, lse,
                                    splits, config.max_splits, config.head_dim ** -0.5)
        return out
    return call


def flashinfer_call(config, inputs, lengths):
    import flashinfer
    q, k, v, pages = inputs
    b = len(lengths)
    counts = [(n + config.page_size - 1) // config.page_size for n in lengths]
    pointers = [0]
    for count in counts:
        pointers.append(pointers[-1] + count)
    indptr = torch.tensor(pointers, device=q.device, dtype=torch.int32)
    indices = torch.cat([pages[r, :count] for r, count in enumerate(counts)])
    tail = torch.tensor([(n-1) % config.page_size + 1 for n in lengths],
                        device=q.device, dtype=torch.int32)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=True)
    def call():
        # Replan is included. In future graph comparisons use the corresponding
        # preallocated graph wrapper and charge per-step metadata update separately.
        wrapper.plan(indptr, indices, tail, config.query_heads, config.kv_heads,
                     config.head_dim, config.page_size, q_data_type=q.dtype, kv_data_type=k.dtype)
        return wrapper.run(q[:b], (k, v))
    return call
