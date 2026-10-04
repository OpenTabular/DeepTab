"""Collection rules for cross-module workflow tests."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    integration_dir = Path(__file__).parent / "integration"
    for item in items:
        if integration_dir in item.path.parents:
            item.add_marker(pytest.mark.integration)
