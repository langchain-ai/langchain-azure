"""Azure Foundry evaluation utilities for LangGraph agents.

This module provides wrappers around Azure Foundry's built-in agent
evaluators and helpers for building evaluator-optimizer subgraphs
in LangGraph workflows.

.. code-block:: python

    from langchain_azure_ai.evaluation import (
        FoundryEvaluator,
        FoundryEvaluatorSuite,
        FoundryEvalResult,
        messages_to_foundry_format,
        tool_schemas_to_foundry_format,
        create_eval_optimize_subgraph,
        create_analyst_subgraph,
    )
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.evaluation.converter import (
        messages_to_foundry_format,
        tool_schemas_to_foundry_format,
    )
    from langchain_azure_ai.evaluation.foundry import (
        FoundryEvalResult,
        FoundryEvaluator,
        FoundryEvaluatorSuite,
    )
    from langchain_azure_ai.evaluation.helpers import (
        create_analyst_subgraph,
        create_eval_optimize_subgraph,
    )

__all__ = [
    "FoundryEvalResult",
    "FoundryEvaluator",
    "FoundryEvaluatorSuite",
    "create_analyst_subgraph",
    "create_eval_optimize_subgraph",
    "messages_to_foundry_format",
    "tool_schemas_to_foundry_format",
]

_module_lookup: dict[str, str] = {
    "FoundryEvalResult": "langchain_azure_ai.evaluation.foundry",
    "FoundryEvaluator": "langchain_azure_ai.evaluation.foundry",
    "FoundryEvaluatorSuite": "langchain_azure_ai.evaluation.foundry",
    "create_analyst_subgraph": "langchain_azure_ai.evaluation.helpers",
    "create_eval_optimize_subgraph": "langchain_azure_ai.evaluation.helpers",
    "messages_to_foundry_format": "langchain_azure_ai.evaluation.converter",
    "tool_schemas_to_foundry_format": "langchain_azure_ai.evaluation.converter",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
