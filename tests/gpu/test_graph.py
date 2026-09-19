import pytest
import torch
from sglang_decode.config import DecodeConfig, Mode
from sglang_decode.runtime.engine import DecodeEngine, PlanKey
from sglang_decode.runtime.graph import DecodeGraph
from sglang_decode.reference.attention import paged_attention
from sglang_decode.planning.policy import PolicyRule
from benchmarks.workloads import make_inputs, split_hints

pytestmark = pytest.mark.gpu


def test_graph_shrink_regrow_switch_modes_and_reset_layers():
    c = DecodeConfig(batch_capacity=32, max_context=1024)
    # Explicit test-only rules exercise a device branch; not production calibration.
    rules = [PolicyRule(1, c.task_capacity, 101, 10000, int(Mode.D4))]
    engine = DecodeEngine(c, layers=2, rules=rules)
    fixtures = [make_inputs(c, [1024]*32, seed=40+i) for i in range(2)]
    key = PlanKey("shared-sequence-order", "test-splits", (32, 8, 128))
    graph = DecodeGraph(engine, [key]*2, [(x[1], x[2]) for x in fixtures])
    pointers = [(x.counter.data_ptr(), x.partial.data_ptr(), x.output.data_ptr()) for x in engine.layers]
    for step, batch in enumerate([32, 7, 1, 32]):
        lengths = [1024 if i % 3 == 0 else 33 for i in range(batch)]
        splits = [1]*batch  # ensure nonuniform per-split work for mode switching
        graph.update([x[0][:batch] for x in fixtures], [x[3][:batch] for x in fixtures], lengths, splits)
        if step == 0:
            graph.capture()
        # Poison everything, not just padding: every valid result must be overwritten.
        for layer in engine.layers:
            layer.partial.fill_(float("nan"))
            layer.lse.fill_(float("nan"))
            layer.counter.fill_(100000)
        outputs = graph.replay()
        for layer, x in enumerate(fixtures):
            expected = paged_attention(*x[:3], x[3], lengths, c.page_size)
            torch.testing.assert_close(outputs[layer][:batch].float(), expected[:batch], atol=2e-2, rtol=2e-2)
        assert pointers == [(x.counter.data_ptr(), x.partial.data_ptr(), x.output.data_ptr()) for x in engine.layers]
        assert engine.plan.mode.item() == (Mode.R1 if batch == 1 else Mode.D4)


def test_capacity_and_layer_compatibility_before_capture():
    c = DecodeConfig(batch_capacity=2, max_context=1024)
    engine = DecodeEngine(c, layers=2)
    x = make_inputs(c, [17, 33])
    first = PlanKey("a", "s", (32, 8, 128))
    second = PlanKey("b", "s", (32, 8, 128))
    with pytest.raises(ValueError):
        DecodeGraph(engine, [first, second], [(x[1], x[2])]*2)
    graph = DecodeGraph(engine, [first]*2, [(x[1], x[2])]*2)
    with pytest.raises(ValueError):
        graph.update([], [], [1, 1, 1], [1, 1, 1])
