"""Tests for the deprecation layer of langchain-azure-dynamic-sessions.

Covers the module-level DeprecationWarning on ``import
langchain_azure_dynamic_sessions`` and the RuntimeWarning emitted by
``_warn_if_deepagents_incompatible`` in ``backends.sessions`` when an
incompatible deepagents version (>= 0.7) is installed.

deepagents is not installed in the test environment, so minimal stand-in
modules are injected into ``sys.modules`` to let ``backends.sessions``
import; ``importlib.metadata.version`` is patched to simulate installed
deepagents versions. The shipped code path is exercised directly (the
version-parsing logic is never reimplemented here).
"""

import importlib
import importlib.metadata
import sys
import types
import warnings
from importlib.metadata import PackageNotFoundError
from typing import Generator, Optional

import pytest

_BACKEND_MODULE = "langchain_azure_dynamic_sessions.backends.sessions"

# Names backends.sessions imports from deepagents at runtime.
_PROTOCOL_NAMES = (
    "EditResult",
    "ExecuteResponse",
    "FileDownloadResponse",
    "FileUploadResponse",
    "WriteResult",
)


def _install_fake_deepagents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject minimal deepagents stand-ins so backends.sessions can import."""
    deepagents = types.ModuleType("deepagents")
    backends = types.ModuleType("deepagents.backends")
    sandbox = types.ModuleType("deepagents.backends.sandbox")
    protocol = types.ModuleType("deepagents.backends.protocol")

    setattr(sandbox, "BaseSandbox", type("BaseSandbox", (), {}))
    for name in _PROTOCOL_NAMES:
        setattr(protocol, name, type(name, (), {}))

    setattr(deepagents, "backends", backends)
    setattr(backends, "sandbox", sandbox)
    setattr(backends, "protocol", protocol)

    monkeypatch.setitem(sys.modules, "deepagents", deepagents)
    monkeypatch.setitem(sys.modules, "deepagents.backends", backends)
    monkeypatch.setitem(sys.modules, "deepagents.backends.sandbox", sandbox)
    monkeypatch.setitem(sys.modules, "deepagents.backends.protocol", protocol)


def _patch_deepagents_version(
    monkeypatch: pytest.MonkeyPatch, installed: Optional[str]
) -> None:
    """Make importlib.metadata report *installed* for deepagents.

    ``None`` simulates deepagents not being installed (PackageNotFoundError).
    Other distributions resolve through the real lookup.
    """
    real_version = importlib.metadata.version

    def fake_version(distribution_name: str) -> str:
        if distribution_name == "deepagents":
            if installed is None:
                raise PackageNotFoundError(distribution_name)
            return installed
        return real_version(distribution_name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)


@pytest.fixture
def evict_backend_modules() -> Generator[None, None, None]:
    """Evict cached backends modules before and after the test."""

    def evict() -> None:
        for key in list(sys.modules):
            if "langchain_azure_dynamic_sessions.backends" in key:
                del sys.modules[key]

    evict()
    yield
    evict()


@pytest.fixture
def sessions_module(
    monkeypatch: pytest.MonkeyPatch,
    evict_backend_modules: None,
) -> types.ModuleType:
    """Import the real backends.sessions module under fake deepagents."""
    _install_fake_deepagents(monkeypatch)
    # Import quietly: simulate deepagents metadata being absent so the
    # module-level compatibility check does not fire during import.
    _patch_deepagents_version(monkeypatch, None)
    return importlib.import_module(_BACKEND_MODULE)


def test_package_import_emits_deprecation_warning() -> None:
    """Importing the package must warn and point at the replacement package."""
    import langchain_azure_dynamic_sessions as pkg

    # Re-execute the module body to re-trigger the module-level warning.
    # reload() reuses the same module object, so no state needs restoring.
    with pytest.warns(DeprecationWarning, match="langchain-azure-container-apps"):
        importlib.reload(pkg)


@pytest.mark.parametrize(
    ("installed", "should_warn"),
    [
        pytest.param("0.6.12", False, id="0.6.12-compatible"),
        pytest.param("0.7.0", True, id="0.7.0"),
        pytest.param("0.7rc1", True, id="0.7rc1-prerelease"),
        pytest.param("0.7.0.dev0", True, id="0.7.0.dev0-prerelease"),
        pytest.param("0.8.1", True, id="0.8.1"),
        pytest.param("1.0.0", True, id="1.0.0"),
        pytest.param("not-a-version", False, id="non-version-string"),
        pytest.param(None, False, id="not-installed"),
    ],
)
def test_warn_if_deepagents_incompatible(
    sessions_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    installed: Optional[str],
    should_warn: bool,
) -> None:
    """RuntimeWarning fires iff the deepagents release tuple is >= (0, 7)."""
    _patch_deepagents_version(monkeypatch, installed)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sessions_module._warn_if_deepagents_incompatible()

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    if should_warn:
        assert len(runtime_warnings) == 1
        message = str(runtime_warnings[0].message)
        assert installed is not None
        assert installed in message
        assert "langchain-azure-container-apps" in message
    else:
        assert runtime_warnings == []


def test_backend_import_warns_on_incompatible_deepagents(
    monkeypatch: pytest.MonkeyPatch,
    evict_backend_modules: None,
) -> None:
    """The compatibility check runs at import time of backends.sessions."""
    _install_fake_deepagents(monkeypatch)
    _patch_deepagents_version(monkeypatch, "0.7.0")

    with pytest.warns(RuntimeWarning, match="deepagents 0.7.0"):
        importlib.import_module(_BACKEND_MODULE)


def test_backend_import_quiet_on_compatible_deepagents(
    monkeypatch: pytest.MonkeyPatch,
    evict_backend_modules: None,
) -> None:
    """No RuntimeWarning when a compatible deepagents (< 0.7) is installed."""
    _install_fake_deepagents(monkeypatch)
    _patch_deepagents_version(monkeypatch, "0.6.12")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module(_BACKEND_MODULE)

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
