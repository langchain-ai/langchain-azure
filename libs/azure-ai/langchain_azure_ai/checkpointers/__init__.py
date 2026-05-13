"""LangGraph checkpoint savers backed by Azure services.

Currently provides:

- :class:`FoundryCheckpointSaver` – persists LangGraph checkpoints to the
  Azure AI Foundry checkpoint storage service.

The Foundry checkpoint storage REST surface is in preview
(api-version ``2025-11-15-preview``); this integration is marked
experimental and may change without a normal deprecation cycle.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.checkpointers._foundry_checkpoint_saver import (
        FoundryCheckpointSaver,
    )

__all__ = [
    "FoundryCheckpointSaver",
]

_module_lookup = {
    "FoundryCheckpointSaver": (
        "langchain_azure_ai.checkpointers._foundry_checkpoint_saver"
    ),
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
