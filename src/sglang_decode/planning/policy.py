"""Offline policy loading. Uncalibrated workloads default to direct round robin."""
import json
from dataclasses import dataclass
from ..config import Mode


@dataclass(frozen=True)
class PolicyRule:
    min_tasks: int
    max_tasks: int
    min_ratio: int  # max/min loop count, ratio scaled by 100
    max_ratio: int
    mode: int


def load_policy(path=None):
    if path is None:
        return []
    data = json.loads(open(path, encoding="utf-8").read())
    if data.get("status") != "validated" or not data.get("evidence"):
        raise ValueError("only explicitly validated policy artifacts may select a dynamic path")
    rules = [PolicyRule(**row) for row in data["rules"]]
    for rule in rules:
        if rule.mode not in (Mode.R1, Mode.STATIC, Mode.D1, Mode.D4):
            raise ValueError("invalid policy mode")
        if rule.min_tasks < 0 or rule.max_tasks < rule.min_tasks or rule.min_ratio < 100:
            raise ValueError("invalid policy bounds")
    return rules
