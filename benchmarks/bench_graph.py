"""Attention-stack graph microbenchmark; this is NOT an entire language model."""
import argparse
import torch
from sglang_decode.config import DecodeConfig, Mode
from sglang_decode.runtime.engine import DecodeEngine, PlanKey
from sglang_decode.runtime.graph import DecodeGraph
from .workloads import WORKLOADS, split_hints, make_inputs
from .common import save_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error("explicit --run-gpu required")
    c = DecodeConfig(workers=torch.cuda.get_device_properties(0).multi_processor_count)
    lengths = WORKLOADS["A"]
    inputs = [make_inputs(c, lengths, seed=17+i) for i in range(args.layers)]
    engine = DecodeEngine(c, layers=args.layers, force_mode=Mode.R1)
    key = PlanKey("A", "fixture", (c.query_heads, c.kv_heads, c.head_dim))
    graph = DecodeGraph(engine, [key] * args.layers, [(x[1], x[2]) for x in inputs])
    graph.update([x[0][:32] for x in inputs], [x[3][:32] for x in inputs], lengths, split_hints(lengths))
    graph.capture()
    rows = []
    for trial in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        rows.append(dict(variant="R1_shared_attention_stack", trial=trial,
                         gpu_us=start.elapsed_time(end)*1000/args.iterations, status="ok"))
    save_run(args.output, rows, dict(layers=args.layers, graph_mode="replay", config=vars(c),
             time_boundary="plan/reset/attention/merge only; excludes model MLP, head and sampling"))


if __name__ == "__main__":
    main()
