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
    _dts_missing = (
        importlib.util.find_spec("durabletask") is None
        or importlib.util.find_spec("durabletask.azuremanaged") is None
    )
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
    CUSTOM_STATUS_INLINE_LIMIT,
    DURABLE_TOOL_ERROR_KEY,
    MAX_GRAPH_NAME_LENGTH,
    RESUME_EVENT_PREFIX,
    STATIC_AFTER_PREFIX,
    STATIC_BEFORE_PREFIX,
    STATIC_INTERRUPT_PREFIX,
    CheckpointBundle,
    Checkpointer,
    ClientProtocol,
    CompiledGraph,
    ExecutionMode,
    FunctionAppHttpServer,
    InterruptEntry,
    LangGraphFunctionApp,
    Limits,
    NodeCallable,
    NodeConfig,
    NodeReport,
    NodeRequest,
    NodeTaskSpec,
    OrchestratorInput,
    OrchestratorOutput,
    OrchestratorState,
    PackedBlob,
    PendingInterrupt,
    PolicyValue,
    Registration,
    Run,
    RunClient,
    RunError,
    Selector,
    StepKind,
    StepReport,
    StepRequest,
    StepStatus,
    TaskResultStatus,
    ToolCallable,
    ToolCallSpec,
    ToolConfig,
    ToolReport,
    ToolRequest,
    WorkerProtocol,
    attach_durability,
    create_function_app,
    dispatch_function_request,
    durable_node,
    durable_tool,
    mark_node,
    mark_tool,
    pack,
    read_output,
    register_function_routes,
    sdk_options,
    stop_function_app,
    unpack,
)

__all__ = [
    "CUSTOM_STATUS_INLINE_LIMIT",
    "CheckpointBundle",
    "Checkpointer",
    "ClientProtocol",
    "CompiledGraph",
    "DURABLE_TOOL_ERROR_KEY",
    "ExecutionMode",
    "FunctionAppHttpServer",
    "InterruptEntry",
    "LangGraphFunctionApp",
    "Limits",
    "MAX_GRAPH_NAME_LENGTH",
    "NodeCallable",
    "NodeConfig",
    "NodeReport",
    "NodeRequest",
    "NodeTaskSpec",
    "OrchestratorInput",
    "OrchestratorOutput",
    "OrchestratorState",
    "PackedBlob",
    "PendingInterrupt",
    "PolicyValue",
    "RESUME_EVENT_PREFIX",
    "Registration",
    "Run",
    "RunClient",
    "RunError",
    "STATIC_AFTER_PREFIX",
    "STATIC_BEFORE_PREFIX",
    "STATIC_INTERRUPT_PREFIX",
    "Selector",
    "StepKind",
    "StepReport",
    "StepRequest",
    "StepStatus",
    "TaskResultStatus",
    "ToolCallSpec",
    "ToolCallable",
    "ToolConfig",
    "ToolReport",
    "ToolRequest",
    "WorkerProtocol",
    "attach_durability",
    "create_function_app",
    "dispatch_function_request",
    "durable_node",
    "durable_tool",
    "mark_node",
    "mark_tool",
    "pack",
    "read_output",
    "register_function_routes",
    "sdk_options",
    "stop_function_app",
    "unpack",
]
