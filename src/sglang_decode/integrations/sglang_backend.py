"""SGLang v0.4.6.post5 adapter (7e257cd...). Prefill stays in upstream.

The model-runner import patch selects this subclass. Unsupported model geometry,
logit caps, speculative modes and capacity overflow keep the original decode.
"""
import os
import torch
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from ..config import DecodeConfig, Mode
from ..planning.policy import load_policy
from ..runtime.engine import DecodeEngine


class ScheduledTritonBackend(TritonAttnBackend):
    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner, **kwargs)
        self.scheduled = None
        self._prepared = False
        self._eligible_step = False
        self.reuse_plan = os.getenv("DECODE_REUSE_PLAN", "1") == "1"
        if os.getenv("DECODE_DISABLE", "0") == "1":
            return
        if getattr(model_runner.server_args, "pp_size", 1) != 1:
            return
        c = model_runner.model_config
        if (getattr(c, "is_multimodal", False) or getattr(c, "is_encoder_decoder", False)
                or getattr(c, "is_deepseek_mla", False)
                or model_runner.server_args.speculative_algorithm is not None):
            return
        key = model_runner.token_to_kv_pool.get_key_buffer(0)
        value = model_runner.token_to_kv_pool.get_value_buffer(0)
        if key.shape != value.shape or key.dtype not in (torch.float16, torch.bfloat16):
            return
        if self.num_head % self.num_kv_head or self.num_head // self.num_kv_head not in (1, 2, 4, 8, 16):
            return
        if key.shape[-1] not in (64, 128, 256):
            return
        config = DecodeConfig(
            batch_capacity=int(os.getenv("DECODE_BATCH_CAPACITY", "64")),
            max_splits=self.max_kv_splits, max_context=self.max_context_len,
            query_heads=self.num_head, kv_heads=self.num_kv_head, head_dim=key.shape[-1],
            page_size=1, workers=max(1, self.device_core_count))
        mode_name = os.getenv("DECODE_MODE", "AUTO")
        force = -1 if mode_name == "AUTO" else int(Mode[mode_name])
        rules = load_policy(os.getenv("DECODE_POLICY"))
        # Indexed by absolute layer_id; pipeline parallel partitions may leave gaps.
        layers = c.hf_config.num_hidden_layers
        self.scheduled = DecodeEngine(config, self.device, key.dtype, layers, rules, force)
        self.query_buffers = [torch.empty_like(x.output) for x in self.scheduled.layers]

    def _begin_step(self, bs, decode, spec_info):
        self._prepared = False
        self._eligible_step = (self.scheduled is not None and decode and spec_info is None
                               and 0 < bs <= self.scheduled.config.batch_capacity)
        if self._eligible_step:
            self.scheduled.plan.active.fill_(bs)

    def init_forward_metadata(self, forward_batch):
        super().init_forward_metadata(forward_batch)
        self._begin_step(forward_batch.batch_size, forward_batch.forward_mode.is_decode_or_idle(),
                         forward_batch.spec_info)

    def init_forward_metadata_capture_cuda_graph(self, bs, num_tokens, req_pool_indices,
            seq_lens, encoder_lens, forward_mode, spec_info):
        super().init_forward_metadata_capture_cuda_graph(
            bs, num_tokens, req_pool_indices, seq_lens, encoder_lens, forward_mode, spec_info)
        self._begin_step(bs, forward_mode.is_decode_or_idle(), spec_info)

    def init_forward_metadata_replay_cuda_graph(self, bs, req_pool_indices, seq_lens,
            seq_lens_sum, encoder_lens, forward_mode, spec_info, seq_lens_cpu):
        super().init_forward_metadata_replay_cuda_graph(
            bs, req_pool_indices, seq_lens, seq_lens_sum, encoder_lens, forward_mode,
            spec_info, seq_lens_cpu)
        # CPU gate chooses a captured bucket; captured nodes rebuild the device plan.
        self._begin_step(bs, forward_mode.is_decode_or_idle(), spec_info)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        # Graph runners may warm up repeatedly after only one metadata hook.
        # Re-arm on the model's first layer so capture always records preparation.
        # Pipeline parallelism is excluded above, so layer 0 is the step boundary.
        if layer.layer_id == 0:
            self._prepared = False
        engine = self.scheduled
        if (not self._eligible_step or layer.logit_cap > 0
                or layer.tp_q_head_num != engine.config.query_heads
                or layer.tp_k_head_num != engine.config.kv_heads
                or layer.qk_head_dim != engine.config.head_dim
                or layer.v_head_dim != engine.config.head_dim
                or getattr(layer, "sliding_window_size", -1) not in (None, -1)):
            return super().forward_decode(q, k, v, layer, forward_batch, save_kv_cache)
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        meta, p = self.forward_metadata, engine.plan
        batch = q.shape[0]
        if not self._prepared or not self.reuse_plan:
            # These device operations are inside the captured model graph, not a
            # Python replay-time branch. Upstream split decisions are copied as-is.
            p.lengths[:batch].copy_(meta.kv_indptr[1:batch + 1] - meta.kv_indptr[:batch])
            p.splits[:batch].copy_(meta.num_kv_splits[:batch])
            engine.prepare()
            self._prepared = True
        qbuf = self.query_buffers[layer.layer_id]
        qbuf[:batch].copy_(q.view(batch, engine.config.query_heads, engine.config.head_dim))
        out = engine.execute(
            qbuf, forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id), meta.kv_indices,
            layer=layer.layer_id, indptr=meta.kv_indptr, scale=layer.scaling)
        return out[:batch].view(batch, -1)
