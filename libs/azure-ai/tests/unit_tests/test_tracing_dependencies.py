"""Tracing dependencies are included in the default V2 installation."""

from importlib.metadata import requires

import pytest
from packaging.requirements import Requirement


@pytest.mark.parametrize(
    "dependency",
    [
        "azure-monitor-opentelemetry",
        "opentelemetry-api",
        "opentelemetry-instrumentation",
        "opentelemetry-instrumentation-threading",
        "opentelemetry-semantic-conventions",
        "opentelemetry-semantic-conventions-ai",
        "wrapt",
    ],
)
def test_tracing_dependency_is_installed_by_default(dependency: str) -> None:
    requirements = [Requirement(item) for item in requires("langchain-azure-ai") or []]
    assert any(item.name == dependency and item.marker is None for item in requirements)
