# Third-party sources and attribution

This repository combines independently written scheduling/runtime code with an
unmodified SGLang baseline and adaptations of published kernel techniques.
References are mechanism sources, not evidence for personal benchmark results.

| Source | Concrete use | License / boundary |
|---|---|---|
| [SGLang v0.4.6.post5](https://github.com/sgl-project/sglang/tree/7e257cd666c0d639626487987ea8e590da1e9395) | `third_party/sglang/decode_attention.py` is an unmodified baseline; 32-token split alignment, normalized partial output/LSE, and backend lifecycle inform local code | Apache-2.0; original header and original LightLLM attribution retained; license included |
| [Triton fused attention tutorial v3.2.0](https://github.com/triton-lang/triton/blob/v3.2.0/python/tutorials/06-fused-attention.py) | Q/K/V tiling, online softmax and FP32 accumulator pattern inform `kernels/attention.py`; local paged/grouped/persistent control flow is rewritten | MIT; license included in `third_party/triton/LICENSE` |
| [FlashInfer paper, v1](https://arxiv.org/html/2501.01005v1) | Plan/execute separation, workload-aware scheduling, plan reuse, fixed graph workspace; `benchmarks/baselines.py` calls the public library API | Conceptual reference, no copied FlashInfer implementation or figures; optional installed library keeps its own license |
| [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) | Capture warmup, stable storage, stream/event ordering in `runtime/graph.py` | API/design reference; wrapper code written here |

The local contribution is the combination of bounded SoA records, stable bucket
ordering, bounded work packs with singleton tails, shared task mapping, dispatch
variants, original-slot writes, compatible-layer plan reuse, and integration.
Neither attention mathematics nor persistent scheduling is claimed as a new
invention. No percentage of borrowed code is invented.

Exact vendored hashes are in `third_party/sources.json`. The pinned integration
version is a reconstruction target, not a claim about the lost environment.
