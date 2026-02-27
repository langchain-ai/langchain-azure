"""**Stores** provide persistent key-value storage backed by Azure AI services.

**Class hierarchy:**

```output
BaseStore --> AzureAIMemoryStore
```
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.stores.azure_ai_memory import (
        AzureAIMemoryStore,
    )

__all__ = [
    "AzureAIMemoryStore",
]

_module_lookup = {
    "AzureAIMemoryStore": "langchain_azure_ai.stores.azure_ai_memory",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
