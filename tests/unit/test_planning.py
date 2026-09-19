"""Specification tests. Written during reconstruction; not executed in this delivery."""
import pytest
from sglang_decode.config import DecodeConfig
from sglang_decode.planning.reference import (
    split_records, bucket_order, make_packs, greedy_static, validate_plan,
)


@pytest.mark.parametrize("lengths,splits,records_count,pack_count", [
    ([512]*24 + [8192]*8, [1]*24 + [16]*8, 152, 1216),
    ([256]*24 + [1536]*4 + [16384]*4, [1]*24 + [2]*4 + [16]*4, 96, 705),
    (([256]*24 + [1536]*4 + [16384]*4)*2, ([1]*24 + [2]*4 + [16]*4)*2, 192, 1329),
])
def test_documented_task_geometry(lengths, splits, records_count, pack_count):
    c = DecodeConfig()
    records, counts, offsets = split_records(lengths, splits, c)
    order = bucket_order(records)
    packs = make_packs(records, order, c.kv_heads, c.workers)
    assert len(records) == records_count == sum(counts) == offsets[-1]
    assert len(packs) == pack_count
    validate_plan(records, order, packs, c.kv_heads)
    longest = max(r.weight for r in records)
    for first, count in packs:
        assert 1 <= count <= 4
        assert sum(records[order[t // c.kv_heads]].weight
                   for t in range(first, first + count)) <= longest


def test_tail_and_zero_lengths_preserve_original_split_slots():
    records, counts, _ = split_records([0, 1, 33, 513], [16, 16, 16, 2], DecodeConfig())
    assert counts == [0, 1, 2, 2]
    assert [(r.begin, r.end) for r in records if r.request == 3] == [(0, 288), (288, 513)]
    assert all(r.begin < r.end for r in records)


def test_small_bucket_remains_singletons():
    c = DecodeConfig()
    records, _, _ = split_records([128, 1024], [1, 1], c)
    packs = make_packs(records, bucket_order(records), c.kv_heads, c.workers)
    assert all(count == 1 for _, count in packs)


def test_static_lists_cover_tasks_and_balance_uniform_work():
    c = DecodeConfig(workers=4)
    records, _, _ = split_records([512]*8, [1]*8, c)
    lists = greedy_static(records, c.kv_heads, c.workers)
    assert sorted(t for worker in lists for t in worker) == list(range(64))
    assert {len(worker) for worker in lists} == {16}


@pytest.mark.parametrize("lengths,splits", [([32769], [1]), ([1], [0]), ([1]*65, [1]*65)])
def test_capacity_rejected(lengths, splits):
    with pytest.raises(ValueError):
        split_records(lengths, splits, DecodeConfig())
