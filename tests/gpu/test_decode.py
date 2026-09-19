import pytest
import torch
from sglang_decode.config import DecodeConfig, Mode
from sglang_decode.runtime.engine import DecodeEngine
from sglang_decode.reference.attention import paged_attention
from sglang_decode.planning.reference import split_records, bucket_order, make_packs
from benchmarks.workloads import make_inputs, update_plan, split_hints

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize("lengths", [[0, 1, 17, 33, 257, 513], [256]*24+[1536]*4+[16384]*4])
def test_outputs_and_gpu_records(mode, lengths):
    c = DecodeConfig()
    engine = DecodeEngine(c, force_mode=mode)
    inputs = make_inputs(c, lengths)
    update_plan(engine, lengths)
    engine.layers[0].partial.fill_(float("nan"))
    engine.layers[0].lse.fill_(float("nan"))
    engine.prepare()
    out = engine.execute(*inputs)
    expected = paged_attention(*inputs[:3], inputs[3], lengths, c.page_size)
    torch.testing.assert_close(out[:len(lengths)].float(), expected[:len(lengths)], atol=2e-2, rtol=2e-2)
    assert engine.plan.error.item() == 0
    records, counts, _ = split_records(lengths, split_hints(lengths), c)
    assert engine.plan.record_count.item() == len(records)
    assert engine.plan.counts[:len(lengths)].tolist() == counts
    if mode != Mode.R0:
        assert engine.plan.request[:len(records)].tolist() == [r.request for r in records]
        assert engine.plan.begin[:len(records)].tolist() == [r.begin for r in records]
    if mode in (Mode.D1, Mode.D4):
        order = bucket_order(records)
        packs = make_packs(records, order, c.kv_heads, c.workers,
                           pack_limit=1 if mode == Mode.D1 else 4)
        assert engine.plan.order[:len(records)].tolist() == order
        count = engine.plan.num_packs.item()
        assert list(zip(engine.plan.pack_first[:count].tolist(),
                        engine.plan.pack_count[:count].tolist())) == packs


def test_noncontiguous_kv_strides():
    c = DecodeConfig()
    lengths = [17, 513]
    q, k, v, pages = make_inputs(c, lengths)
    def strided(x):
        storage = torch.empty((*x.shape[:-1], x.shape[-1]*2), device=x.device, dtype=x.dtype)
        storage[..., ::2].copy_(x)
        return storage[..., ::2]
    k, v = strided(k), strided(v)
    engine = DecodeEngine(c, force_mode=Mode.D4)
    update_plan(engine, lengths)
    engine.prepare()
    out = engine.execute(q, k, v, pages)
    expected = paged_attention(q, k, v, pages, lengths, c.page_size)
    torch.testing.assert_close(out[:2].float(), expected[:2], atol=2e-2, rtol=2e-2)
