# Copyright (c) Microsoft. All rights reserved.

"""Azure Durable Task Scheduler (DTS) durable execution for LangGraph graphs.

Provides :func:`attach_durability`, :func:`durable_tool`, :func:`durable_node`,
and :func:`sdk_options` to run compiled LangGraph graphs on Azure Durable Task
Scheduler or the local DTS emulator. Requires the optional ``durabletask``
extra::

    pip install "langchain-azure-compute[durabletask]"
"""

import importlib.util

if importlib.util.find_spec("langgraph") is None:
    raise ImportError(
        "The Azure Durable Task Scheduler LangGraph integration requires "
        "langgraph, which is provided by the 'durabletask' extra. Install "
        "it with: pip install "
        "'langchain-azure-compute[durabletask]'."
    )

try:
    _dts_missing = importlib.util.find_spec("durabletask.azuremanaged") is None
except ModuleNotFoundError:
    _dts_missing = True

if _dts_missing:
    raise ImportError(
        "The Azure Durable Task Scheduler LangGraph integration requires "
        "durabletask and durabletask-azuremanaged, which are provided by the "
        "'durabletask' extra. Install it with: pip install "
        "'langchain-azure-compute[durabletask]'."
    )

from langchain_azure_compute.graph._api import (
    AsyncRun,
    AsyncRunClient,
    ExecutionMode,
    Limits,
    PendingInterrupt,
    Registration,
    Run,
    RunClient,
    RunError,
    attach_durability,
    durable_node,
    durable_tool,
    sdk_options,
)

__all__ = [
    "AsyncRun",
    "AsyncRunClient",
    "ExecutionMode",
    "Limits",
    "PendingInterrupt",
    "Registration",
    "Run",
    "RunClient",
    "RunError",
    "attach_durability",
    "durable_node",
    "durable_tool",
    "sdk_options",
]
