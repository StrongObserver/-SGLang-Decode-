"""Real SGLang model generation harness; reports whole-generation wall time.

This does not substitute for the historical fixed-snapshot decode-step metric.
Launch each variant in a fresh process using the same pinned model and prompts.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--model", required=True, help="local immutable model snapshot")
    parser.add_argument("--token-ids", required=True, type=Path, help="JSON list of prompt token lists")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["upstream", "R0", "R1", "STATIC", "D1", "D4"], default="R1")
    parser.add_argument("--reuse-plan", choices=["0", "1"], default="1")
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error("model loading and execution require --run-gpu")
    os.environ["DECODE_DISABLE"] = "1" if args.mode == "upstream" else "0"
    os.environ["DECODE_MODE"] = "R1" if args.mode == "upstream" else args.mode
    os.environ["DECODE_REUSE_PLAN"] = args.reuse_plan
    from sglang import Engine
    inputs = json.loads(args.token_ids.read_text())
    engine = Engine(model_path=args.model, attention_backend="triton", dtype="bfloat16",
                    disable_cuda_graph=args.disable_cuda_graph, tp_size=1,
                    triton_attention_num_kv_splits=16)
    sampling = dict(temperature=0, max_new_tokens=args.new_tokens, ignore_eos=True)
    rows = []
    try:
        engine.generate(input_ids=inputs, sampling_params=sampling)
        for trial in range(5):
            engine.flush_cache()
            begin = time.perf_counter_ns()
            output = engine.generate(input_ids=inputs, sampling_params=sampling)
            elapsed = time.perf_counter_ns() - begin
            rows.append(dict(trial=trial, elapsed_ms=elapsed / 1e6,
                             meta_info=[item["meta_info"] for item in output]))
    finally:
        engine.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(dict(variant=args.mode, reuse_plan=args.reuse_plan, model=args.model,
                       graph=not args.disable_cuda_graph, samples=rows,
                       time_boundary="host whole generation: prefill + decode + output; NOT fixed-step GPU time"),
                  f, indent=2)


if __name__ == "__main__":
    main()
