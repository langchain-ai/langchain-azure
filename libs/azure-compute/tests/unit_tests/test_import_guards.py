"""Every module ships in the wheel; the extras only gate dependencies.

So each subpackage checks its own dependencies at import time and raises an
error naming the extra to install. These tests re-execute a module with
`importlib.util.find_spec` stubbed to report a dependency missing, which is the
only way to reach those branches in an environment where it is installed.

The module is executed into a throwaway object rather than reloaded in place,
so a half-executed module never lands in `sys.modules`.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from unittest import mock

import pytest

_REAL_FIND_SPEC = importlib.util.find_spec


def _exec_module_without(module_name: str, missing: str) -> ModuleType:
    """Execute `module_name` fresh, with `missing` reported as not installed."""

    def fake_find_spec(name: str, package: str | None = None) -> object | None:
        return None if name == missing else _REAL_FIND_SPEC(name, package)

    spec = _REAL_FIND_SPEC(module_name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch("importlib.util.find_spec", side_effect=fake_find_spec):
        spec.loader.exec_module(module)
    return module


class TestDynamicSessionsPackage:
    def test_missing_requests_names_the_extra(self) -> None:
        with pytest.raises(ImportError, match="dynamic-sessions") as excinfo:
            _exec_module_without("langchain_azure_compute.dynamic_sessions", "requests")
        assert "requests" in str(excinfo.value)


def test_find_spec_is_restored_after_each_stub() -> None:
    assert importlib.util.find_spec is _REAL_FIND_SPEC


def test_user_agent_falls_back_when_the_distribution_is_absent() -> None:
    """The package can be run from a source tree with no installed metadata."""
    import importlib.metadata

    spec = _REAL_FIND_SPEC("langchain_azure_compute.dynamic_sessions._common")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(
        importlib.metadata,
        "version",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
        spec.loader.exec_module(module)
    assert module.USER_AGENT == "langchain-azure-compute/0.0.0 (Language=Python)"
