"""Fixed-grid plan launches; no allocation, host readback or tensor rebinding."""
import triton
from ..kernels.plan import count_scan, emit_records, make_order_and_packs, greedy_lists


def build_plan(plan, force_mode=-1):
    c = plan.config
    count_scan[(1,)](plan.lengths, plan.splits, plan.active, plan.counts, plan.offsets,
                    plan.chunk, plan.record_count, plan.error, c.batch_capacity,
                    c.max_splits, c.max_context, triton.next_power_of_2(c.batch_capacity))
    if int(force_mode) == 0:
        plan.mode.fill_(0)
        plan.num_packs.zero_()
        return
    emit_records[(c.batch_capacity,)](
        plan.lengths, plan.counts, plan.offsets, plan.chunk, plan.request, plan.split,
        plan.begin, plan.end, plan.weight, c.max_splits, c.block_kv,
        triton.next_power_of_2(c.max_splits))
    make_order_and_packs[(1,)](
        plan.weight, plan.record_count, plan.order, plan.pack_first, plan.pack_count,
        plan.num_packs, plan.mode, plan.rules, plan.rule_count, int(force_mode),
        c.kv_heads, c.workers, c.buckets, triton.next_power_of_2(c.record_capacity),
        triton.next_power_of_2(c.task_capacity), num_warps=4)
    if int(force_mode) == 2 or any_static_rule(plan):
        greedy_lists[(1,)](plan.weight, plan.record_count, plan.mode, plan.static_head,
                          plan.static_next, c.kv_heads, c.workers,
                          triton.next_power_of_2(c.workers),
                          triton.next_power_of_2(c.record_capacity), num_warps=4)


def any_static_rule(plan):
    # Capture-time metadata set by the runtime, never inspect a device scalar here.
    return getattr(plan, "has_static_rule", False)
