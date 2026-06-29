"""LangChain integrations for Azure Storage."""

import importlib.util
from importlib import metadata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_storage._deepagents import (
        AzureBlobBackend,
        AzureBlobConfig,
    )

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

# Names re-exported lazily from the optional ``[deepagents]`` extra. They are
# loaded on first access (PEP 562) so importing ``langchain_azure_storage``
# stays cheap and does not require ``deepagents`` (which pulls in a large
# dependency tree) unless the backend is actually used.
_DEEPAGENTS_EXPORTS = {"AzureBlobBackend", "AzureBlobConfig"}

__all__ = ["AzureBlobBackend", "AzureBlobConfig"]


def __getattr__(name: str) -> Any:
    """Lazily import optional, extra-gated public names (PEP 562)."""
    if name in _DEEPAGENTS_EXPORTS:
        if importlib.util.find_spec("deepagents") is None:
            raise ImportError(
                f"{name} requires the 'deepagents' extra. Install it with: "
                "pip install 'langchain-azure-storage[deepagents]' "
                "(requires Python >= 3.11)."
            )
        # deepagents is installed: import directly so genuine import errors
        # (e.g. a dependency-version conflict) surface instead of being masked
        # as a missing-extra message.
        from langchain_azure_storage import _deepagents

        return getattr(_deepagents, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exported names in ``dir()``."""
    return sorted(set(globals()) | _DEEPAGENTS_EXPORTS)
