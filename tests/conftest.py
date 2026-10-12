"""Collection rules for workflow categories and dependency profiles."""

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--dependency-profile",
        choices=("core", "extras"),
        default=None,
        help="Verify the externally provisioned core-only or logs/tracking installation.",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    root = Path(__file__).parent
    categories = {
        root / "integration": "integration",
        root / "contracts": "contract",
        root / "learnability": "learnability",
        root / "lifecycle": "lifecycle",
        root / "metrics" / "oracles": "metric_oracle",
        root / "boundaries": "boundary",
        root / "environments": "environment",
    }
    for item in items:
        for directory, marker in categories.items():
            if directory in item.path.parents:
                item.add_marker(getattr(pytest.mark, marker))
