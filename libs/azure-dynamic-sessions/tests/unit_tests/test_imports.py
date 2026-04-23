import importlib
import sys

import pytest

from langchain_azure_dynamic_sessions import __all__

EXPECTED_ALL = [
    "SessionsBashTool",
    "SessionsPythonREPLTool",
]

# deepagents sub-modules to block when simulating the package being absent
_DEEPAGENTS_MODULES = (
    "deepagents",
    "deepagents.backends",
    "deepagents.backends.sandbox",
    "deepagents.backends.protocol",
)


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_main_package_imports_without_deepagents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main package classes must be importable even when deepagents is absent."""
    # Block deepagents sub-modules so any import attempt raises ImportError
    for mod in _DEEPAGENTS_MODULES:
        monkeypatch.setitem(sys.modules, mod, None)  # type: ignore[arg-type]

    # Remove any cached sessions/backends modules so they would be re-imported
    for key in list(sys.modules):
        if "langchain_azure_dynamic_sessions.backends" in key:
            monkeypatch.delitem(sys.modules, key)

    # The main __init__ must succeed: it only imports from tools.sessions
    import langchain_azure_dynamic_sessions as pkg

    assert hasattr(pkg, "SessionsBashTool")
    assert hasattr(pkg, "SessionsPythonREPLTool")


def test_backends_import_raises_friendly_error_without_deepagents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing backends.sessions without deepagents must raise a clear ImportError."""
    # Block deepagents
    for mod in _DEEPAGENTS_MODULES:
        monkeypatch.setitem(sys.modules, mod, None)  # type: ignore[arg-type]

    # Evict cached sessions/backends modules so they are freshly imported
    for key in list(sys.modules):
        if "langchain_azure_dynamic_sessions.backends" in key:
            monkeypatch.delitem(sys.modules, key)

    with pytest.raises(ImportError, match="deepagents"):
        importlib.import_module("langchain_azure_dynamic_sessions.backends.sessions")
