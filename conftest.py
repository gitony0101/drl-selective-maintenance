import pytest


def pytest_ignore_collect(collection_path, config):
    path_str = str(collection_path).replace("\\", "/")
    if path_str.endswith(
        "evidence/heldout/final_test_evaluation/run_core_m6_test.py"
    ):
        return True
    return False