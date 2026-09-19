"""CPU specification and strong host-side static baseline; no GPU dependency."""
from dataclasses import dataclass
import heapq


@dataclass(frozen=True)
class Record:
    request: int
    split: int
    begin: int
    end: int
    weight: int


def split_records(lengths, requested_splits, config):
    if len(lengths) != len(requested_splits) or len(lengths) > config.batch_capacity:
        raise ValueError("batch/split shape exceeds the capture contract")
    records, offsets, counts = [], [0], []
    for request, (length, splits) in enumerate(zip(lengths, requested_splits)):
        if not 0 <= length <= config.max_context or not 1 <= splits <= config.max_splits:
            raise ValueError("invalid length or split count")
        chunk = ((length + splits - 1) // splits + 31) // 32 * 32
        count = (length + chunk - 1) // chunk if chunk else 0
        counts.append(count)
        for split in range(count):
            begin, end = split * chunk, min((split + 1) * chunk, length)
            records.append(Record(request, split, begin, end,
                                  (end - begin + config.block_kv - 1) // config.block_kv))
        offsets.append(len(records))
    return records, counts, offsets


def bucket_order(records):
    return sorted(range(len(records)), key=lambda i: -(records[i].weight - 1).bit_length())


def make_packs(records, order, groups, workers, pack_limit=4):
    """Pack within one bucket. Reserve a full worker wave of singleton tasks."""
    packs = []
    max_weight = max((r.weight for r in records), default=1)
    first = 0
    while first < len(order):
        bucket = (records[order[first]].weight - 1).bit_length()
        end = first + 1
        while end < len(order) and (records[order[end]].weight - 1).bit_length() == bucket:
            end += 1
        tasks = (end - first) * groups
        width = min(pack_limit, max(1, max_weight // (1 << bucket)))
        packed = max(0, tasks - workers) // width * width
        start = first * groups
        packs.extend((start + i, width) for i in range(0, packed, width))
        packs.extend((start + i, 1) for i in range(packed, tasks))
        first = end
    return packs


def greedy_static(records, groups, workers):
    """LPT with O(T log workers) heap assignment when host lengths already exist."""
    tasks = sorted(range(len(records) * groups), key=lambda t: (-records[t // groups].weight, t))
    heap = [(0, c) for c in range(workers)]
    heapq.heapify(heap)
    lists = [[] for _ in range(workers)]
    for task in tasks:
        load, worker = heapq.heappop(heap)
        lists[worker].append(task)
        heapq.heappush(heap, (load + records[task // groups].weight, worker))
    return lists


def validate_plan(records, order, packs, groups):
    assert sorted(order) == list(range(len(records)))
    tasks = [task for first, count in packs for task in range(first, first + count)]
    assert sorted(tasks) == list(range(len(records) * groups))
    slots = {(records[order[t // groups]].request, records[order[t // groups]].split,
              t % groups) for t in tasks}
    assert len(slots) == len(tasks)
