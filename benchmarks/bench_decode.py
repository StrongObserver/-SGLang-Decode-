"""Single-layer full-entry ablation. Explicit --run-gpu is required."""
import argparse
import statistics
import time
import torch
from sglang_decode.config import DecodeConfig, Mode
from sglang_decode.runtime.engine import DecodeEngine
from sglang_decode.planning.reference import split_records, greedy_static
from sglang_decode.reference.attention import paged_attention
from .workloads import WORKLOADS, make_inputs, update_plan, split_hints
from .common import save_run


def prepare_host_static(engine, lengths):
    """No D2H: scheduler already owns lengths. Charge heap planning and H2D upload."""
    records, _, _ = split_records(lengths, split_hints(lengths), engine.config)
    lists = greedy_static(records, engine.config.kv_heads, engine.config.workers)
    heads = [-1] * engine.config.workers
    links = [-1] * engine.config.task_capacity
    for worker, tasks in enumerate(lists):
        if tasks:
            heads[worker] = tasks[0]
        for left, right in zip(tasks, tasks[1:]):
            links[left] = right
    # R1 produces the common records, then replace only dispatch metadata.
    engine.prepare()
    p = engine.plan
    p.static_head.copy_(torch.tensor(heads, dtype=torch.int32, device=p.device))
    p.static_next.copy_(torch.tensor(links, dtype=torch.int32, device=p.device))
    p.mode.fill_(int(Mode.STATIC))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=WORKLOADS, default="B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--external", nargs="*", choices=["upstream", "flashinfer"], default=[])
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error("this entry compiles/executes kernels; opt in with --run-gpu")
    c = DecodeConfig(workers=torch.cuda.get_device_properties(0).multi_processor_count)
    lengths = WORKLOADS[args.workload]
    inputs = make_inputs(c, lengths)
    expected = paged_attention(*inputs[:3], inputs[3], lengths, c.page_size)
    variants = {name: DecodeEngine(c, force_mode=int(mode)) for name, mode in
                [("R0", Mode.R0), ("R1", Mode.R1), ("STATIC_GPU", Mode.STATIC),
                 ("D1", Mode.D1), ("D4", Mode.D4), ("STATIC_CPU", Mode.R1)]}
    calls = {}
    for name, engine in variants.items():
        update_plan(engine, lengths)
        def call(name=name, engine=engine):
            if name == "STATIC_CPU":
                prepare_host_static(engine, lengths)
            else:
                engine.prepare()
            return engine.execute(*inputs)
        calls[name] = call
        actual = call()
        torch.testing.assert_close(actual[:len(lengths)].float(), expected[:len(lengths)],
                                   atol=2e-2, rtol=2e-2)
        for _ in range(args.warmup):
            call()
    from .baselines import upstream_call, flashinfer_call
    for name in args.external:
        factory = upstream_call if name == "upstream" else flashinfer_call
        call = factory(c, inputs, lengths)
        torch.testing.assert_close(call().float(), expected[:len(lengths)], atol=2e-2, rtol=2e-2)
        for _ in range(args.warmup):
            call()
        calls[name] = call
    torch.cuda.synchronize()
    rows = []
    for trial in range(args.rounds):
        names = list(calls) if trial % 2 == 0 else list(reversed(calls))
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter_ns()
            start.record()
            for _ in range(args.iterations):
                calls[name]()
            end.record()
            end.synchronize()
            wall_us = (time.perf_counter_ns() - wall_start) / 1000 / args.iterations
            rows.append(dict(variant=name, trial=trial, iterations=args.iterations,
                             gpu_us=start.elapsed_time(end) * 1000 / args.iterations,
                             wall_us=wall_us, status="ok"))
    save_run(args.output, rows, dict(workload=args.workload, lengths=lengths,
             config=vars(c), warmup=args.warmup, repeats=args.rounds, graph_mode="eager",
             time_boundary="plan begin through reset/attention/merge; wall includes host LPT"))
    for name in calls:
        print(name, "median full-entry wall us:", statistics.median(
            r["wall_us"] for r in rows if r["variant"] == name))


if __name__ == "__main__":
    main()
