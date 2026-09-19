"""GPU tests require both pytest -m gpu and DECODE_RUN_GPU_TESTS=1."""
import os
import pytest


def pytest_collection_modifyitems(items):
    if os.getenv("DECODE_RUN_GPU_TESTS") != "1":
        for item in items:
            item.add_marker(pytest.mark.skip(reason="explicit GPU execution opt-in required"))
