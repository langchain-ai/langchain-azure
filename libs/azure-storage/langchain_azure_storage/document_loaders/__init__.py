"""Document Loaders are classes to load Documents.

Document Loaders are usually used to load a lot of Documents in a single run.

Class hierarchy:

.. code-block::

    BaseLoader --> <name>Loader  # Examples: TextLoader, UnstructuredFileLoader

Main helpers:

.. code-block::

    Document, <name>TextSplitter
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseLoader
    from .unstructured import (
        UnstructuredAPIFileIOLoader,
        UnstructuredAPIFileLoader,
        UnstructuredFileIOLoader,
        UnstructuredFileLoader,
    )

_module_lookup = {
    "BaseLoader": ".base",
    "UnstructuredFileLoader": ".unstructured",
    "UnstructuredFileIOLoader": ".unstructured",
    "UnstructuredAPIFileLoader": ".unstructured",
    "UnstructuredAPIFileIOLoader": ".unstructured",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "BaseLoader",
    "UnstructuredFileLoader",
    "UnstructuredFileIOLoader",
    "UnstructuredAPIFileLoader",
    "UnstructuredAPIFileIOLoader",
]
