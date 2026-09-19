import json
import pytest
from sglang_decode.planning.policy import load_policy


def test_uncalibrated_defaults_to_no_rules():
    assert load_policy() == []


def test_unvalidated_artifact_cannot_enable_dynamic(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"status": "unvalidated", "evidence": [], "rules": []}))
    with pytest.raises(ValueError):
        load_policy(path)
