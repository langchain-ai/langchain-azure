# Copyright (c) Microsoft. All rights reserved.

"""Implementation of the public API for durable LangGraph execution on DTS.

This package connects a compiled LangGraph `Pregel` graph to Azure Durable Task
Scheduler
with a clean separation between worker registration and client run management:

  1. **`attach_durability(graph, name=...)` -> `Registration`:**
     Validates the compiled graph and `name` (`MAX_GRAPH_NAME_LENGTH = 64`), configures
     its checkpointer
     (defaulting to zero-config `InMemorySaver` bundle mode when no persistent
     checkpointer is supplied),
     resolves per-tool and per-node durability policies (`ToolConfig`, `NodeConfig`),
     and binds the four
     human-readable DTS task names (`langgraph.<name>.{run,step,tool,node}`).
  2. **`Registration.register(worker)`:**
     Registers the orchestrator (`make_orchestrator`) and activity handlers (`run_step`,
     `run_tool`, `run_node`) onto a `DurableTaskSchedulerWorker`.
  3. **`Registration.client(sdk)` -> `RunClient`:**
     Wraps a `DurableTaskSchedulerClient` to start new runs or attach to existing runs
     idempotently (`start`, `attach_run`).
  4. **`Run` & `PendingInterrupt`:**
     High-level handle for inspecting orchestration status, polling or listing
     human-in-the-loop
     interrupts (`pending_interrupt`, `pending_interrupts`, `wait_pending`), sending
     resume
     events (`resume`), and awaiting final graph output (`wait_result`).

Internal modules:
  - `_config.py`: `Limits`, `RunError`, `ToolConfig`, `@durable_tool`, `pack` /
    `unpack`.
  - `_async_bridge.py`: Persistent background event loop for `AsyncSqliteSaver` / async
    tools.
  - `_inline_saver.py`: Two-tier state model (`InMemorySaver` bundle export/hydration
    vs. pointer-only).
  - `_checkpoints.py`: Composite `<uuid>@<hash>` checkpoint IDs, mid-start crash
    detection,
    parallel `put_writes`, and isolated `PregelScratchpad` tool execution.
  - `runner.py`: Core `_report`, `run_tool`, `run_step`, and `make_orchestrator` engine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from durabletask import task
from durabletask.grpc_options import GrpcChannelOptions
from durabletask.payload.store import PayloadStore
from langchain_core._api import beta
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_azure_compute.graph._checkpoints import _aget_state, _project_output
from langchain_azure_compute.graph._config import (
    _check_payload,
    _configure_msgpack_modules,
)
from langchain_azure_compute.graph._inline_saver import _get_cached_inline_bundle
from langchain_azure_compute.graph.azure_functions import (
    FunctionAppHttpServer,
    LangGraphFunctionApp,
    create_function_app,
    dispatch_function_request,
    register_function_routes,
    stop_function_app,
)
from langchain_azure_compute.graph.runner import (
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
    InterruptEntry,
    Limits,
    MigrationKey,
    MigrationSpec,
    NodeCallable,
    NodeConfig,
    NodeReport,
    NodeRequest,
    NodeTaskSpec,
    OrchestratorInput,
    OrchestratorOutput,
    OrchestratorState,
    PackedBlob,
    PolicyValue,
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
    _checkpoint_id,
    _config,
    _export_checkpoint_bundle,
    _get_state,
    _hydrate_checkpoint_bundle,
    _normalize_run_config,
    _pack_context,
    _prepare_subgraph_tool_nodes,
    _resolve_graph,
    _validate_graph_name,
    durable_node,
    durable_tool,
    make_orchestrator,
    mark_node,
    mark_tool,
    pack,
    read_output,
    run_node,
    run_step,
    run_tool,
    unpack,
)

__all__ = [
    "attach_durability",
    "Registration",
    "RunClient",
    "Run",
    "PendingInterrupt",
    "RunError",
    "Limits",
    "ToolConfig",
    "NodeConfig",
    "CUSTOM_STATUS_INLINE_LIMIT",
    "MAX_GRAPH_NAME_LENGTH",
    "RESUME_EVENT_PREFIX",
    "STATIC_INTERRUPT_PREFIX",
    "STATIC_BEFORE_PREFIX",
    "STATIC_AFTER_PREFIX",
    "DURABLE_TOOL_ERROR_KEY",
    "durable_tool",
    "mark_tool",
    "durable_node",
    "mark_node",
    "pack",
    "unpack",
    "read_output",
    "sdk_options",
    "LangGraphFunctionApp",
    "create_function_app",
    "register_function_routes",
    "dispatch_function_request",
    "FunctionAppHttpServer",
    "stop_function_app",
    "CompiledGraph",
    "Checkpointer",
    "ExecutionMode",
    "PolicyValue",
    "ToolCallable",
    "NodeCallable",
    "Selector",
    "PackedBlob",
    "CheckpointBundle",
    "InterruptEntry",
    "ToolCallSpec",
    "NodeTaskSpec",
    "StepKind",
    "StepStatus",
    "TaskResultStatus",
    "StepRequest",
    "StepReport",
    "ToolRequest",
    "ToolReport",
    "NodeRequest",
    "NodeReport",
    "OrchestratorInput",
    "OrchestratorOutput",
    "OrchestratorState",
    "WorkerProtocol",
    "ClientProtocol",
]

_TERMINAL = {"COMPLETED", "FAILED", "TERMINATED"}
_REUSABLE_STATUSES = {"RUNNING", "PENDING", "SUSPENDED", "COMPLETED"}
_MISSING = object()
_BETA_MESSAGE = (
    "The Azure Durable Task Scheduler LangGraph integration is in public preview. "
    "Its API is not stable and may change in future versions."
)


def _status_name(meta: Any) -> str:
    status = getattr(meta, "runtime_status", None)
    return str(getattr(status, "name", status))


def _default_instance_id(name: str, thread: str, expect: str | None, input: Any) -> str:
    digest = hashlib.sha256(
        json.dumps([name, thread, expect, input], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return "r-" + digest[:32]


def _extract_hostname(host: str) -> str:
    h = host.strip().lower()
    if h.startswith("[") and "]" in h:
        return h[1 : h.index("]")]
    if ":" in h:
        return h.split(":", 1)[0]
    return h


def _is_loopback_host(host: str) -> bool:
    return _extract_hostname(host) in (
        "localhost",
        "127.0.0.1",
        "::1",
    )


def _strip_scheme(endpoint: str) -> tuple[str, bool | None]:
    raw = endpoint.strip()
    low = raw.lower()
    for proto in ("https://", "grpcs://"):
        if low.startswith(proto):
            return raw[len(proto) :].rstrip("/"), True
    for proto in ("http://", "grpc://"):
        if low.startswith(proto):
            return raw[len(proto) :].rstrip("/"), False
    return raw.rstrip("/"), None


@beta(message=_BETA_MESSAGE)
def sdk_options(
    endpoint: str | None = None,
    *,
    taskhub: str | None = None,
    token_credential: Any = None,
    secure_channel: bool | None = None,
    connection_string: str | None = None,
    max_message_bytes: int | None = None,
    payload_store: PayloadStore | None = None,
    log_handler: logging.Handler | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for `DurableTaskSchedulerClient` and.

    `DurableTaskSchedulerWorker`.

    Supports both the local DTS emulator (`"127.0.0.1:8080"`) and hosted Azure Durable
    Task Scheduler
    endpoints (`"https://<scheduler>.<region>.durabletask.io"` with Entra ID
    `DefaultAzureCredential`
    or a `DURABLE_TASK_SCHEDULER_CONNECTION_STRING` of the form
    `Endpoint=...;TaskHub=...;Authentication=...`).
    Optionally raises gRPC receive/send message limits via `max_message_bytes` and
    wires the Durable Task SDK's `PayloadStore` / `BlobPayloadStore`.

    Args:
        endpoint: Emulator or hosted scheduler endpoint; defaults to DTS environment
        settings.
        taskhub: Task hub name, overriding the connection string and environment.
        token_credential: Azure credential used for authenticated endpoints. Defaults
            to `DefaultAzureCredential` for hosted `.durabletask.io` endpoints when
            neither a credential nor a connection-string authentication mode is set.
        secure_channel: Whether to use TLS; inferred from the endpoint when omitted.
        connection_string: Scheduler connection string, optionally read from the
        environment.
        max_message_bytes: Optional maximum gRPC message size; None keeps SDK defaults.
        payload_store: Optional SDK payload storage; this does not bypass the
            integration's wire-size guard. Pass external blob references for
            oversized durable activity data.
        log_handler: Optional logging handler passed to the SDK.

    Returns:
        Keyword arguments accepted by the scheduler client and worker.

    Raises:
        ValueError: If no endpoint is supplied or the connection settings are invalid.
    """
    if (
        connection_string is None
        and isinstance(endpoint, str)
        and "endpoint=" in endpoint.lower()
    ):
        connection_string = endpoint
        endpoint = None
    if endpoint is None and connection_string is None:
        env_conn = os.environ.get(
            "DURABLE_TASK_SCHEDULER_CONNECTION_STRING"
        ) or os.environ.get("DTS_CONNECTION_STRING")
        env_ep = os.environ.get("DTS_ENDPOINT") or os.environ.get("DTS_EMULATOR")
        if env_conn:
            connection_string = env_conn
        elif env_ep:
            if "endpoint=" in env_ep.lower():
                connection_string = env_ep
            elif env_ep.strip().lower() in ("1", "true", "yes"):
                endpoint = "127.0.0.1:8080"
            else:
                endpoint = env_ep

    parsed_endpoint: str | None = None
    parsed_taskhub: str | None = None
    parsed_auth: str | None = None
    parsed_client_id: str | None = None
    if connection_string:
        for part in connection_string.split(";"):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k_low, v_clean = k.strip().lower(), v.strip()
            if k_low == "endpoint":
                parsed_endpoint = v_clean
            elif k_low in ("taskhub", "taskhubname"):
                parsed_taskhub = v_clean
            elif k_low == "authentication":
                parsed_auth = v_clean
            elif k_low == "clientid":
                parsed_client_id = v_clean

    raw_endpoint = (
        endpoint
        or parsed_endpoint
        or os.environ.get("DTS_ENDPOINT")
        or (
            "127.0.0.1:8080"
            if (os.environ.get("DTS_EMULATOR") or "").strip().lower()
            in ("1", "true", "yes")
            else os.environ.get("DTS_EMULATOR")
        )
    )
    if not raw_endpoint:
        raise ValueError("endpoint or connection_string is required")

    host_address, scheme_secure = _strip_scheme(raw_endpoint)
    resolved_taskhub = (
        taskhub
        or parsed_taskhub
        or os.environ.get("DTS_TASKHUB")
        or os.environ.get("TASKHUB_NAME")
        or "default"
    )

    resolved_cred = token_credential
    if parsed_auth is not None:
        auth_low = parsed_auth.lower()
        if auth_low in ("none", "anonymous", "emulator", ""):
            pass
        elif auth_low == "managedidentity":
            if resolved_cred is None:
                from azure.identity import ManagedIdentityCredential

                resolved_cred = (
                    ManagedIdentityCredential(client_id=parsed_client_id)
                    if parsed_client_id
                    else ManagedIdentityCredential()
                )
        elif auth_low == "workloadidentity":
            if resolved_cred is None:
                from azure.identity import WorkloadIdentityCredential

                resolved_cred = (
                    WorkloadIdentityCredential(client_id=parsed_client_id)
                    if parsed_client_id
                    else WorkloadIdentityCredential()
                )
        elif auth_low == "environment":
            if resolved_cred is None:
                from azure.identity import EnvironmentCredential

                resolved_cred = EnvironmentCredential()
        elif auth_low == "azurecli":
            if resolved_cred is None:
                from azure.identity import AzureCliCredential

                resolved_cred = AzureCliCredential()
        elif auth_low == "azurepowershell":
            if resolved_cred is None:
                from azure.identity import AzurePowerShellCredential

                resolved_cred = AzurePowerShellCredential()
        elif auth_low in (
            "defaultazure",
            "defaultazurecredential",
            "entra",
            "entradefault",
            "aad",
        ):
            if resolved_cred is None:
                from azure.identity import DefaultAzureCredential

                resolved_cred = (
                    DefaultAzureCredential(managed_identity_client_id=parsed_client_id)
                    if parsed_client_id
                    else DefaultAzureCredential()
                )
        else:
            raise ValueError(f"unsupported Authentication mode {parsed_auth!r}")

    if (
        resolved_cred is None
        and parsed_auth is None
        and _extract_hostname(host_address).endswith(".durabletask.io")
    ):
        from azure.identity import DefaultAzureCredential

        resolved_cred = DefaultAzureCredential()

    if secure_channel is not None:
        resolved_secure = bool(secure_channel)
    elif scheme_secure is not None:
        resolved_secure = (
            True if (scheme_secure or resolved_cred is not None) else False
        )
    elif resolved_cred is not None:
        resolved_secure = True
    elif _extract_hostname(host_address).endswith(".durabletask.io"):
        resolved_secure = True
    else:
        resolved_secure = False

    if (
        resolved_cred is not None
        and not resolved_secure
        and not _is_loopback_host(host_address)
    ):
        raise ValueError(
            "Refusing to send token_credential over an insecure channel "
            "(secure_channel=False) to a non-loopback host"
        )

    ua_pair = ("grpc.primary_user_agent", "langgraph-durabletask")
    options: dict[str, Any] = dict(
        host_address=host_address,
        taskhub=resolved_taskhub,
        token_credential=resolved_cred,
        secure_channel=resolved_secure,
        channel_options=GrpcChannelOptions(
            max_receive_message_length=(
                int(max_message_bytes) if max_message_bytes is not None else None
            ),
            max_send_message_length=(
                int(max_message_bytes) if max_message_bytes is not None else None
            ),
            raw_options=[ua_pair],
        ),
    )
    if payload_store is not None:
        options["payload_store"] = payload_store
    if log_handler is not None:
        options["log_handler"] = log_handler
    return options


@beta(message=_BETA_MESSAGE)
class PendingInterrupt:
    """Receipt representing an active LangGraph `interrupt()` or static breakpoint.

    awaiting `Run.resume()`.

    Attributes:
        interrupt_id: Deterministic LangGraph interrupt ID (or synthetic static
        breakpoint ID)
            used to route the external event `"resume:<interrupt_id>"`.
        value: Deserialized payload passed to `interrupt(value)` inside the graph node
        or tool.
        step: Orchestration step counter at which the interrupt occurred.
        ckpt: Composite checkpoint identifier at which the interrupt occurred.
    """

    def __init__(
        self,
        interrupt_id: str,
        value: Any,
        *,
        step: int | None = None,
        ckpt: str | None = None,
    ) -> None:
        """Store an interrupt receipt and its optional checkpoint coordinates."""
        self.interrupt_id, self.value = interrupt_id, value
        self.step, self.ckpt = step, ckpt

    def __repr__(self) -> str:
        """Return a compact description of the interrupt receipt."""
        return f"PendingInterrupt({self.interrupt_id[:8]}, {self.value!r})"  # noqa: E501


@beta(message=_BETA_MESSAGE)
class Registration:
    """Binds a compiled LangGraph graph to a named Durable Task orchestrator and.

    activity pair.

    Created via `attach_durability(graph, name=...)`. Use `register(worker)` on the
    worker side and
    `client(sdk)` on the caller side.
    """

    def __init__(
        self,
        graph: CompiledGraph,
        *,
        name: str,
        checkpointer: Checkpointer | None = None,
        limits: Limits,
        tool_config: ToolConfig,
        node_config: NodeConfig | None = None,
        inline_checkpoints: bool | None = None,
        version: str | None = None,
        previous_versions: Mapping[str, CompiledGraph] | None = None,
        migrations: Mapping[MigrationKey, MigrationSpec] | None = None,
        upgrade_on_continue_as_new: bool = False,
        allowed_msgpack_modules: Iterable[tuple[str, ...]] | bool | None = None,
    ) -> None:
        """Bind a compiled graph, task names, and execution policies."""
        import copy

        _configure_msgpack_modules(allowed_msgpack_modules)
        name = _validate_graph_name(name)
        if not hasattr(graph, "get_state") and hasattr(graph, "compile"):
            raise ValueError(
                "graph is an uncompiled StateGraph builder; "
                "call .compile() before attach_durability()"
            )
        orig_graph = graph
        orig_cp = getattr(graph, "checkpointer", None)
        if checkpointer is None and orig_cp is False:
            raise ValueError(
                "graph was compiled with checkpointer=False; "
                "omit checkpointer or provide a checkpointer"
            )
        graph = copy.copy(graph)
        if isinstance(getattr(graph, "interrupt_before_nodes", None), list):
            graph.interrupt_before_nodes = list(graph.interrupt_before_nodes)
        if isinstance(getattr(graph, "interrupt_after_nodes", None), list):
            graph.interrupt_after_nodes = list(graph.interrupt_after_nodes)
        if checkpointer is not None:
            graph.checkpointer = checkpointer
        cp = getattr(graph, "checkpointer", None)
        if cp is None:
            if inline_checkpoints is False:
                raise ValueError(
                    "graph must have a checkpointer when inline_checkpoints=False"
                )
            from langchain_azure_compute.graph._config import _SERDE

            cp = InMemorySaver(serde=_SERDE)
            graph.checkpointer = cp
        if inline_checkpoints is None:
            inline_checkpoints = isinstance(cp, InMemorySaver)
        self.inline_checkpoints = bool(inline_checkpoints)
        ncfg = (
            node_config
            if node_config is not None
            else NodeConfig(default_durable=False)
        )
        self.graph, self.name, self.limits, self.tool_config, self.node_config = (
            graph,
            name,
            limits,
            tool_config,
            ncfg,
        )
        self.version = str(version) if version is not None else None
        self.previous_versions: dict[str, CompiledGraph] = {
            str(k): copy.copy(v) for k, v in (previous_versions or {}).items()
        }
        self.migrations: dict[MigrationKey, MigrationSpec] = dict(migrations or {})
        self.upgrade_on_continue_as_new = bool(upgrade_on_continue_as_new)

        for ver_key, prev_g in self.previous_versions.items():
            if not hasattr(prev_g, "get_state") and hasattr(prev_g, "compile"):
                raise ValueError(
                    f"previous_versions[{ver_key!r}] is an uncompiled "
                    "StateGraph builder; call .compile()"
                )
            if getattr(prev_g, "checkpointer", None) is None:
                prev_g.checkpointer = cp
            setattr(prev_g, "_durable_name", self.name)
            self._configure_single_graph(
                prev_g, tool_config, ncfg, overwrite_config=False
            )

        self._configure_single_graph(
            self.graph, tool_config, ncfg, overwrite_config=True
        )
        setattr(self.graph, "_durable_name", self.name)
        setattr(self.graph, "_durable_version", self.version)
        setattr(self.graph, "_durable_previous_versions", self.previous_versions)
        setattr(self.graph, "_durable_migrations", self.migrations)
        setattr(self.graph, "_durable_upgrade_on_can", self.upgrade_on_continue_as_new)
        from langchain_azure_compute.graph._checkpoints import _register_graph_clone

        _register_graph_clone(orig_graph, self.graph)
        self.orchestrator_name = f"langgraph.{name}.run"
        self.step_name = f"langgraph.{name}.step"
        self.tool_name = f"langgraph.{name}.tool"
        self.node_name = f"langgraph.{name}.node"
        self._active_thread_instances: dict[str, str] = {}

    def _resolve_graph(self, version: str | None) -> CompiledGraph:
        return _resolve_graph(self.graph, version)

    def _configure_single_graph(
        self,
        g: CompiledGraph,
        tool_config: ToolConfig,
        node_config: NodeConfig,
        *,
        overwrite_config: bool = True,
    ) -> None:
        if overwrite_config or not hasattr(g, "_durable_tool_config"):
            setattr(g, "_durable_tool_config", tool_config)
        if overwrite_config or not hasattr(g, "_durable_node_config"):
            setattr(g, "_durable_node_config", node_config)
        setattr(g, "_durable_inline_checkpoints", self.inline_checkpoints)
        setattr(g, "_durable_async_timeout", self.limits.async_timeout_seconds)
        if not hasattr(g, "_durable_orig_interrupt_before"):
            ub = getattr(g, "interrupt_before_nodes", None)
            setattr(
                g,
                "_durable_orig_interrupt_before",
                "*" if ub == "*" else tuple(ub or ()),
            )
        if not hasattr(g, "_durable_orig_interrupt_after"):
            ua = getattr(g, "interrupt_after_nodes", None)
            setattr(
                g,
                "_durable_orig_interrupt_after",
                "*" if ua == "*" else tuple(ua or ()),
            )
        orig_ub = getattr(g, "_durable_orig_interrupt_before")
        orig_ua = getattr(g, "_durable_orig_interrupt_after")
        setattr(g, "_durable_user_interrupt_before", orig_ub)
        setattr(g, "_durable_user_interrupt_after", orig_ua)
        g.interrupt_before_nodes = "*" if orig_ub == "*" else list(orig_ub)
        g.interrupt_after_nodes = "*" if orig_ua == "*" else list(orig_ua)
        _prepare_subgraph_tool_nodes(
            g, self.limits, getattr(g, "_durable_tool_config", tool_config)
        )

    def register(self, worker: WorkerProtocol) -> None:
        """Register the graph's orchestrator and step, tool, and node activities.

        Args:
            worker: Durable Task worker receiving the graph's handlers.
        """
        orchestrator = make_orchestrator(self.limits, name=self.orchestrator_name)
        worker_logger = getattr(worker, "_logger", None)

        def _inst_id(ctx: task.ActivityContext) -> str:
            val = getattr(ctx, "orchestration_id", None) or getattr(
                ctx, "instance_id", None
            )
            return str(val) if val is not None else "unknown"

        def step(
            ctx: task.ActivityContext, request: StepRequest | dict[str, Any]
        ) -> dict[str, Any]:
            req_ver = request.get("version") if isinstance(request, Mapping) else None
            kind = (
                str(request.get("kind", "step"))
                if isinstance(request, Mapping)
                else "step"
            )
            thread = (
                str(request.get("thread", "")) if isinstance(request, Mapping) else ""
            )
            inst = _inst_id(ctx)
            request = dict(request)
            if kind == "start":
                request.setdefault("instance_id", inst)
            target_g = _resolve_graph(self.graph, req_ver)
            tcfg = getattr(target_g, "_durable_tool_config", self.tool_config)
            ncfg = getattr(target_g, "_durable_node_config", self.node_config)
            pre_next: tuple[str, ...] = ()
            if worker_logger is not None and thread and kind in ("step", "resume"):
                try:
                    if (
                        isinstance(request, Mapping)
                        and "checkpoint_bundle" in request
                        and isinstance(
                            getattr(target_g, "checkpointer", None), InMemorySaver
                        )
                    ):
                        _hydrate_checkpoint_bundle(
                            target_g,
                            thread,
                            request["checkpoint_bundle"],
                            only_if_missing_cfg=_config(thread),
                        )
                    pre_next = tuple(
                        getattr(_get_state(target_g, _config(thread)), "next", ()) or ()
                    )
                except Exception:
                    pre_next = ()
            try:
                rep = run_step(
                    self.graph, self.limits, request, tool_config=tcfg, node_config=ncfg
                )
            except RunError as exc:
                # Returning a failure envelope avoids SDK activity retries for
                # deterministic conflicts and payload-limit violations.
                if worker_logger is not None:
                    worker_logger.warning(
                        f"{inst}: Activity {self.step_name} (kind={kind}) failed: {exc}"
                    )
                return {"status": StepStatus.ERROR, "error": str(exc)}
            except Exception as exc:
                if worker_logger is not None:
                    worker_logger.warning(
                        f"{inst}: Activity {self.step_name} (kind={kind}) "
                        f"failed: {type(exc).__name__}: {exc}"
                    )
                raise
            if worker_logger is not None:
                st = rep.get("status", "unknown")
                post_next: list[str] = []
                ran_nodes: list[str] = list(pre_next)
                if thread:
                    try:
                        post_snap = _get_state(target_g, _config(thread))
                        post_next = [
                            str(n) for n in (getattr(post_snap, "next", ()) or ())
                        ]
                        if not ran_nodes and kind == "start":
                            cp_saver = getattr(target_g, "checkpointer", None)
                            tup = (
                                cp_saver.get_tuple(_config(thread))
                                if cp_saver is not None
                                and hasattr(cp_saver, "get_tuple")
                                else None
                            )
                            vseen = (
                                tup.checkpoint.get("versions_seen", {})
                                if tup is not None and getattr(tup, "checkpoint", None)
                                else {}
                            )
                            ran_nodes = [
                                str(k) for k in vseen if not str(k).startswith("__")
                            ]
                    except Exception:
                        pass
                ran_info = ""
                if kind == "tools":
                    ran_info = f", committed={request.get('node') or 'tools'!r}"
                elif kind == "nodes":
                    ran_info = ", committed='nodes'"
                elif len(ran_nodes) == 1:
                    ran_info = f", node={ran_nodes[0]!r}"
                elif ran_nodes:
                    ran_info = f", nodes={ran_nodes}"
                extra = ""
                if st == "tools":
                    names = [tc.get("name", "?") for tc in rep.get("tool_calls", ())]
                    extra = f", next={rep.get('node')!r}, tools={names}"
                elif st == "nodes":
                    names = [nt.get("name", "?") for nt in rep.get("node_tasks", ())]
                    extra = f", next={names}"
                elif post_next:
                    extra = (
                        f", next={post_next[0]!r}"
                        if len(post_next) == 1
                        else f", next={post_next}"
                    )
                worker_logger.info(
                    f"{inst}: Activity {self.step_name} "
                    f"(kind={kind}{ran_info}) completed -> status={st}{extra}"
                )
            return rep

        def tool_activity(
            ctx: task.ActivityContext, request: ToolRequest | dict[str, Any]
        ) -> dict[str, Any]:
            inst = _inst_id(ctx)
            t_name = "?"
            if isinstance(request, Mapping) and "call" in request:
                try:
                    t_name = unpack(request["call"]).get("name", "?")
                except Exception:
                    t_name = "?"
            try:
                rep = run_tool(self.graph, request)
            except RunError as exc:
                return {"status": StepStatus.ERROR, "error": str(exc)}
            except Exception as exc:
                if worker_logger is not None:
                    worker_logger.warning(
                        f"{inst}: Activity {self.tool_name} (tool={t_name!r}) "
                        f"failed: {type(exc).__name__}: {exc}"
                    )
                if getattr(exc, "_durable_tool_error_info", None) is not None:
                    from langchain_azure_compute.graph.runner import (
                        _extract_failure_info,
                    )

                    raise RuntimeError(_extract_failure_info(exc)["message"]) from exc
                raise
            if worker_logger is not None:
                worker_logger.info(
                    f"{inst}: Activity {self.tool_name} (tool={t_name!r}) "
                    f"completed -> status={rep.get('status', 'ok')}"
                )
            return rep

        def node_activity(
            ctx: task.ActivityContext, request: NodeRequest | dict[str, Any]
        ) -> dict[str, Any]:
            inst = _inst_id(ctx)
            n_name = request.get("name", "?") if isinstance(request, Mapping) else "?"
            try:
                rep = run_node(self.graph, request)
            except RunError as exc:
                return {"status": StepStatus.ERROR, "error": str(exc)}
            except Exception as exc:
                if worker_logger is not None:
                    worker_logger.warning(
                        f"{inst}: Activity {self.node_name} (node={n_name!r}) "
                        f"failed: {type(exc).__name__}: {exc}"
                    )
                raise
            if worker_logger is not None:
                worker_logger.info(
                    f"{inst}: Activity {self.node_name} (node={n_name!r}) "
                    f"completed -> status={rep.get('status', 'ok')}"
                )
            return rep

        step.__name__ = self.step_name
        tool_activity.__name__ = self.tool_name
        node_activity.__name__ = self.node_name
        worker.add_orchestrator(orchestrator)
        worker.add_activity(step)
        worker.add_activity(tool_activity)
        worker.add_activity(node_activity)

        from langchain_azure_compute.graph._config import _versioned_activity_name

        supported_versions: list[str] = []
        if self.version is not None:
            supported_versions.append(self.version)
        for prev_v in self.previous_versions:
            if prev_v not in supported_versions:
                supported_versions.append(prev_v)
        for ver in supported_versions:

            def _make_ver_handlers(
                v: str,
            ) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
                def v_step(
                    ctx: task.ActivityContext,
                    request: StepRequest | dict[str, Any],
                ) -> dict[str, Any]:
                    return step(ctx, request)

                def v_tool(
                    ctx: task.ActivityContext,
                    request: ToolRequest | dict[str, Any],
                ) -> dict[str, Any]:
                    return tool_activity(ctx, request)

                def v_node(
                    ctx: task.ActivityContext,
                    request: NodeRequest | dict[str, Any],
                ) -> dict[str, Any]:
                    return node_activity(ctx, request)

                v_step.__name__ = _versioned_activity_name(self.step_name, v)
                v_tool.__name__ = _versioned_activity_name(self.tool_name, v)
                v_node.__name__ = _versioned_activity_name(self.node_name, v)
                return v_step, v_tool, v_node

            vs, vt, vn = _make_ver_handlers(ver)
            worker.add_activity(vs)
            worker.add_activity(vt)
            worker.add_activity(vn)
        if (
            hasattr(worker, "use_work_item_filters")
            and not getattr(worker, "_is_running", False)
            and getattr(worker, "_work_item_filters", None) is None
        ):
            try:
                worker.use_work_item_filters()
            except Exception:
                pass

    def client(self, sdk: ClientProtocol) -> RunClient:
        """Create a `RunClient` bound to this registration and the given DTS client.

        `sdk`.
        """
        return RunClient(self, sdk)

    def async_client(self, sdk: Any) -> AsyncRunClient:
        """Create an `AsyncRunClient` bound to this registration and an async DTS client."""  # noqa: E501
        return AsyncRunClient(self, sdk)

    def register_function_routes(
        self,
        app: Any,
        sdk_or_options: Any = None,
        *,
        start_worker: bool = False,
        identity_provider: Callable[[Any], str | None] | None = None,
        require_instance_owner: bool | None = None,
        respond_authorizer: Callable[..., bool] | None = None,
    ) -> Any:
        """Register Azure Functions v2 HTTP routes for this graph.

        Args:
            app: Function app receiving start, status, respond, and health routes.
            sdk_or_options: Existing DTS client or SDK connection options.
            start_worker: Whether to start an embedded worker.
            identity_provider: Callback validating tokens and returning principal IDs.
            require_instance_owner: Enforce per-user ownership; see create_function_app.
            respond_authorizer: Optional approver policy (caller, ID, owner[, graph]).
                Authorized approvers may also read the instance's status.

        Returns:
            The supplied function app with graph routes registered.
        """
        return register_function_routes(
            app,
            self,
            sdk_or_options,
            start_worker=start_worker,
            identity_provider=identity_provider,
            require_instance_owner=require_instance_owner,
            respond_authorizer=respond_authorizer,
        )

    def create_function_app(
        self,
        dts_options: Any = None,
        *,
        auth_level: Any = None,
        start_worker: bool = False,
        identity_provider: Callable[[Any], str | None] | None = None,
        require_instance_owner: bool | None = None,
        respond_authorizer: Callable[..., bool] | None = None,
    ) -> Any:
        """Create thin HTTP routes; embedded workers require local/Always-On hosting."""
        return create_function_app(
            self,
            dts_options,
            auth_level=auth_level,
            start_worker=start_worker,
            identity_provider=identity_provider,
            require_instance_owner=require_instance_owner,
            respond_authorizer=respond_authorizer,
        )


def _existing_run_input(existing: Any, owner: str | None) -> dict[str, Any]:
    """Read durable routing metadata and fail closed on owner mismatch."""
    try:
        payload = json.loads(getattr(existing, "serialized_input", None) or "{}")
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if owner is not None and payload.get("owner") != owner:
        raise PermissionError("Existing orchestration belongs to another owner")
    return payload


def _default_thread(instance_id: str | None, owner: str | None) -> str:
    """Keep owner-scoped checkpoint keys distinct even after DTS instance purge."""
    thread = instance_id or "t-" + uuid.uuid4().hex
    if owner is not None:
        owner_hash = hashlib.sha256(owner.encode("utf-8")).hexdigest()
        return f"owner:{owner_hash}:{thread}"
    return thread


def _check_checkpoint_owner(snapshot: Any, owner: str | None) -> None:
    """Reject retained checkpoints belonging to a different principal."""
    metadata = getattr(snapshot, "metadata", None) or {}
    checkpoint_owner = metadata.get("durable_owner")
    if checkpoint_owner is not None and checkpoint_owner != owner:
        raise PermissionError("Checkpoint thread belongs to another owner")


@beta(message=_BETA_MESSAGE)
def attach_durability(
    graph: CompiledGraph,
    *,
    name: str,
    checkpointer: Checkpointer | None = None,
    limits: Limits | None = None,
    durable_tools: bool
    | ExecutionMode
    | str
    | Iterable[Selector]
    | Mapping[Selector, PolicyValue] = True,
    tool_policy: Mapping[Selector, PolicyValue] | None = None,
    tool_retries: int | Mapping[Selector, int] | None = None,
    tool_max_concurrency: int | None = None,
    max_concurrency: int | None = None,
    durable_nodes: bool
    | ExecutionMode
    | str
    | Iterable[Selector]
    | Mapping[Selector, PolicyValue] = False,
    node_policy: Mapping[Selector, PolicyValue] | None = None,
    node_retries: int | Mapping[Selector, int] | None = None,
    node_max_concurrency: int | None = None,
    inline_checkpoints: bool | None = None,
    version: str | None = None,
    previous_versions: Mapping[str, CompiledGraph] | None = None,
    migrations: Mapping[MigrationKey, MigrationSpec] | None = None,
    upgrade_on_continue_as_new: bool = False,
    allowed_msgpack_modules: Iterable[tuple[str, ...]] | bool | None = None,
) -> Registration:
    """Attach a compiled LangGraph `graph` to Durable Task under namespace `name`.

    Args:
        graph: A compiled LangGraph `Pregel` graph (`.compile()` must already have been
        called).
        name: Logical name (`<= MAX_GRAPH_NAME_LENGTH` chars) used to namespace the DTS
        orchestrator and activities (`langgraph.<name>.*`).
        checkpointer: Optional checkpointer override (`SqliteSaver`, `AsyncSqliteSaver`,
        `InMemorySaver`, etc.).
            If omitted and the graph was compiled without a checkpointer, an
            `InMemorySaver` is attached
            automatically in zero-config inline-bundle mode.
        limits: Optional `Limits` governing step caps, `ContinueAsNew` rollover, and
        retry counts.
        durable_tools: Controls which tools in `ToolNode`s run as isolated DTS
        activities (`True` by default;
            accepts a bool, iterable of tool names/objects, or mapping of tool selectors
            to policies).
        tool_policy: Optional mapping of tool or `"node::tool"` selectors to `"durable"`
        / `"local"` / retry count.
        tool_retries: Optional default retry count (or per-tool mapping) for durable
        tool activities.
        tool_max_concurrency: Optional concurrency limit for tool calls in a `ToolNode`
        superstep (`None` = fan out
            all in parallel in one batch, `1` = sequential, `N > 1` = bounded batches of
            size `N`).
        max_concurrency: Alias for `tool_max_concurrency`.
        durable_nodes: Controls which graph nodes (including parallel fan-out branches
        and `Send()` targets) run as
            isolated `langgraph.<name>.node` activities (`False` by default, so only
            `@durable_node`-decorated nodes
            fan out as activities; set `True` to run all regular nodes as separate
            activities).
        node_policy: Optional mapping of node names/callables to `"durable"` / `"local"`
        / retry count.
        node_retries: Optional default retry count (or per-node mapping) for durable
        node activities.
        node_max_concurrency: Optional concurrency limit for durable node tasks in a
        superstep (`None` = fan out all
            in parallel in one batch, `1` = sequential, `N > 1` = bounded batches of
            size `N`).
        inline_checkpoints: Whether to ship `InMemorySaver` checkpoint slices through
        DTS payloads.
            Defaults to `True` for `InMemorySaver` and `False` for persistent
            checkpointers.
        version: Optional version tag (e.g. `"v2"`) for this graph definition.
        previous_versions: Optional mapping of `{version_str: compiled_graph}` so
        in-flight runs started on older
            graph versions continue executing safely against their original topology
            until migrated.
        migrations: Optional mapping of `(from_version, to_version)` (or `"v1->v2"` /
        `"v2"`) to a migration
            callable `fn(state, snapshot[, resume_val])` or dict `{"values": ...,
            "as_node": ...}`.
        upgrade_on_continue_as_new: When `True`, runs on an older version automatically
        migrate to `version`
            when rolling over via `continue_as_new`.
        allowed_msgpack_modules: Explicitly trusted msgpack modules or types registered
            process-wide for custom state values. True permits all types; use only
            with trusted payloads. None keeps the restricted defaults.

    Returns:
        Registration for worker handlers and client run management.

    Raises:
        ValueError: If the graph name or durability configuration is invalid.
    """
    tcfg = ToolConfig.from_options(
        durable_tools=durable_tools,
        tool_policy=tool_policy,
        tool_retries=tool_retries,
        max_concurrency=max_concurrency,
        tool_max_concurrency=tool_max_concurrency,
    )
    ncfg = NodeConfig.from_options(
        durable_nodes=durable_nodes,
        node_policy=node_policy,
        node_retries=node_retries,
        node_max_concurrency=node_max_concurrency,
    )
    return Registration(
        graph,
        name=name,
        checkpointer=checkpointer,
        limits=limits or Limits(),
        tool_config=tcfg,
        node_config=ncfg,
        inline_checkpoints=inline_checkpoints,
        version=version,
        previous_versions=previous_versions,
        migrations=migrations,
        upgrade_on_continue_as_new=upgrade_on_continue_as_new,
        allowed_msgpack_modules=allowed_msgpack_modules,
    )


@beta(message=_BETA_MESSAGE)
class RunClient:
    """Client factory for starting or re-attaching to durable LangGraph runs."""

    def __init__(self, registration: Registration, sdk: ClientProtocol) -> None:
        """Bind a graph registration to a Durable Task client."""
        self.registration, self.sdk = registration, sdk

    def start(
        self,
        input: Any,
        *,
        thread_id: str | None = None,
        instance_id: str | None = None,
        reuse_existing: bool = True,
        max_concurrency: int | None = None,
        node_max_concurrency: int | None = None,
        config: RunnableConfig | Mapping[str, Any] | None = None,
        context: Any = None,
        version: str | None = None,
        owner: str | None = None,
    ) -> Run:
        """Start or idempotently attach to a durable LangGraph run.

        Args:
            input: Initial graph input, or None to continue an existing thread.
            thread_id: Checkpointer conversation identifier. Without an instance ID,
                each turn gets a fresh orchestration ID.
            instance_id: Execution identifier; default thread ID is owner-namespaced
                when owner is supplied.
            reuse_existing: Attach rather than schedule again when the instance is
                RUNNING, PENDING, SUSPENDED, or COMPLETED.
            max_concurrency: Tool fan-out limit; None uses the registration default.
            node_max_concurrency: Durable node fan-out limit for this run.
            config: Runnable configuration forwarded to nodes and durable tools.
            context: LangGraph runtime context forwarded to nodes and durable tools.
            version: Graph version, defaulting to the registration's version.
            owner: Optional caller principal ID recorded durably for route ownership.

        Returns:
            A handle to the new or existing orchestration.

        Raises:
            ValueError: If a concurrency limit is less than one.
        """
        reg = self.registration
        cfg_dict = dict(config) if isinstance(config, Mapping) else {}
        user_conf = (
            dict(cfg_dict.get("configurable") or {})
            if isinstance(cfg_dict.get("configurable"), Mapping)
            else {}
        )
        if max_concurrency is None and cfg_dict.get("max_concurrency") is not None:
            max_concurrency = int(cfg_dict["max_concurrency"])
        eff_mc = (
            reg.tool_config.max_concurrency
            if max_concurrency is None
            else int(max_concurrency)
        )
        if eff_mc is not None and eff_mc < 1:
            raise ValueError("max_concurrency must be >= 1 or None")
        eff_nmc = (
            reg.node_config.max_concurrency
            if node_max_concurrency is None
            else int(node_max_concurrency)
        )
        if eff_nmc is not None and eff_nmc < 1:
            raise ValueError("node_max_concurrency must be >= 1 or None")
        eff_ver = str(version) if version is not None else reg.version
        target_graph = _resolve_graph(reg.graph, eff_ver)
        explicit_thread = thread_id or (
            str(user_conf["thread_id"]) if user_conf.get("thread_id") else None
        )
        if (
            instance_id is not None
            and reuse_existing
            and hasattr(self.sdk, "get_orchestration_state")
        ):
            existing = self.sdk.get_orchestration_state(
                instance_id, fetch_payloads=False
            )
            status_obj = (
                getattr(existing, "runtime_status", None)
                if existing is not None
                else None
            )
            status_name = getattr(status_obj, "name", status_obj)
            if status_name in _REUSABLE_STATUSES:
                if not getattr(existing, "serialized_input", None):
                    existing = self.sdk.get_orchestration_state(
                        instance_id, fetch_payloads=True
                    )
                existing_input = _existing_run_input(existing, owner)
                thread = existing_input.get("thread") or explicit_thread
                if thread:
                    _check_checkpoint_owner(
                        _get_state(target_graph, _config(thread)), owner
                    )
                return Run(reg, self.sdk, instance_id, thread or instance_id)
        thread = explicit_thread or _default_thread(instance_id, owner)
        if (
            explicit_thread is not None
            and instance_id is None
            and thread in reg._active_thread_instances
            and hasattr(self.sdk, "get_orchestration_state")
        ):
            prev_iid = reg._active_thread_instances[thread]
            prev_state = self.sdk.get_orchestration_state(
                prev_iid, fetch_payloads=False
            )
            prev_st_obj = (
                getattr(prev_state, "runtime_status", None)
                if prev_state is not None
                else None
            )
            prev_st = getattr(prev_st_obj, "name", prev_st_obj)
            if prev_st in ("RUNNING", "PENDING", "SUSPENDED"):
                raise RunError(
                    f"thread {thread!r} already has an active run ({prev_iid!r}, "
                    f"status={prev_st}); wait for completion or pass an explicit instance_id"  # noqa: E501
                )
        norm_cfg = _normalize_run_config(cfg_dict)
        snapshot = _get_state(target_graph, _config(thread, run_config=norm_cfg))
        _check_checkpoint_owner(snapshot, owner)
        expect = _checkpoint_id(snapshot)
        if reg.inline_checkpoints and expect is None and input is None:
            cached = _get_cached_inline_bundle(target_graph, thread)
            if cached is not None:
                _hydrate_checkpoint_bundle(target_graph, thread, cached)
                snapshot = _get_state(
                    target_graph, _config(thread, run_config=norm_cfg)
                )
                _check_checkpoint_owner(snapshot, owner)
                expect = _checkpoint_id(snapshot)
        if input is None and expect is None:
            raise ValueError("cannot start an empty thread without checkpoint state")
        request: dict[str, Any] = {
            "thread": thread,
            "kind": StepKind.START,
            "expect": expect,
            "input": pack(input),
        }
        iid = instance_id or _default_instance_id(
            reg.name, thread, expect, request["input"]
        )
        request["instance_id"] = iid
        if owner is not None:
            request["owner"] = owner
        if eff_ver is not None:
            request["version"] = eff_ver
        if eff_mc is not None:
            request["max_concurrency"] = eff_mc
        if eff_nmc is not None:
            request["node_max_concurrency"] = eff_nmc
        if norm_cfg is not None:
            request["run_config"] = pack(norm_cfg)
        if context is not None:
            request["context"] = _pack_context(context)
        if reg.inline_checkpoints and expect is not None:
            bundle = _export_checkpoint_bundle(target_graph, thread)
            if bundle is not None:
                request["checkpoint_bundle"] = bundle
        payload: dict[str, Any] = {
            "proto": 1,
            "thread": thread,
            "gen": 0,
            "steps": 0,
            "req": request,
            "step_name": reg.step_name,
            "tool_name": reg.tool_name,
            "node_name": reg.node_name,
            "limits": {
                "custom_status_limit": reg.limits.custom_status_limit,
                "max_steps": reg.limits.max_steps,
                "steps_per_generation": reg.limits.steps_per_generation,
                "byte_budget": reg.limits.byte_budget,
                "retries": reg.limits.retries,
                "tool_retries": reg.limits.tool_retries,
                "node_retries": reg.limits.node_retries,
                "async_timeout_seconds": reg.limits.async_timeout_seconds,
            },
        }
        if eff_ver is not None:
            payload["version"] = eff_ver
        if reg.upgrade_on_continue_as_new:
            payload["upgrade_on_can"] = True
            if reg.version is not None:
                payload["target_version"] = reg.version
        if eff_mc is not None:
            payload["max_concurrency"] = eff_mc
        if eff_nmc is not None:
            payload["node_max_concurrency"] = eff_nmc
        if "run_config" in request:
            payload["run_config"] = request["run_config"]
        if "context" in request:
            payload["context"] = request["context"]
        if owner is not None:
            payload["owner"] = str(owner)
        _check_payload(payload, "orchestrator input")
        self.sdk.schedule_new_orchestration(
            reg.orchestrator_name, input=payload, instance_id=iid
        )
        if len(reg._active_thread_instances) >= 256:
            reg._active_thread_instances.pop(
                next(iter(reg._active_thread_instances)), None
            )
        reg._active_thread_instances[thread] = iid
        return Run(reg, self.sdk, iid, thread)

    def attach_run(self, instance_id: str, thread_id: str | None = None) -> Run:
        """Attach a `Run` handle to an existing orchestration `instance_id` (resolving.

        `thread_id` from DTS if omitted).
        """
        if thread_id is None and hasattr(self.sdk, "get_orchestration_state"):
            existing = self.sdk.get_orchestration_state(
                instance_id, fetch_payloads=True
            )
            if existing is not None:
                for raw_field in (
                    "serialized_input",
                    "serialized_output",
                ):
                    raw_val = getattr(existing, raw_field, None)
                    if raw_val:
                        try:
                            parsed = json.loads(raw_val)
                            if isinstance(parsed, Mapping) and parsed.get("thread"):
                                thread_id = str(parsed["thread"])
                                break
                        except Exception:
                            pass
        return Run(self.registration, self.sdk, instance_id, thread_id or instance_id)


@beta(message=_BETA_MESSAGE)
class Run:
    """Handle to an in-flight or completed durable LangGraph orchestration."""

    def __init__(
        self,
        registration: Registration,
        sdk: ClientProtocol,
        instance_id: str,
        thread_id: str,
    ) -> None:
        """Create a run handle for the orchestration and checkpointer thread."""
        self._reg, self._sdk = registration, sdk
        self.instance_id, self.thread_id = instance_id, thread_id
        self.generations: int = 0
        self.steps: int = 0
        self.version: str | None = None

    def status(self) -> str:
        """Return the DTS orchestration runtime status name (`RUNNING`, `COMPLETED`.

        `FAILED`, etc.).
        """
        state = self._sdk.get_orchestration_state(
            self.instance_id, fetch_payloads=False
        )
        return "NOT_FOUND" if state is None else str(state.runtime_status.name)

    def _resolve_interrupt(
        self,
        entry: Mapping[str, Any],
        *,
        step: int | None = None,
        ckpt: str | None = None,
    ) -> PendingInterrupt:
        """Hydrate a `PendingInterrupt` from custom status or fall back to reading.

        large values from the checkpointer.
        """
        iid = entry["id"]
        if "value" in entry:
            return PendingInterrupt(iid, unpack(entry["value"]), step=step, ckpt=ckpt)
        snap = _get_state(self._reg.graph, _config(self.thread_id))
        for t in snap.tasks:
            for intr in t.interrupts:
                if intr.id == iid:
                    return PendingInterrupt(
                        iid, intr.value, step=step, ckpt=ckpt or _checkpoint_id(snap)
                    )
        return PendingInterrupt(iid, None, step=step, ckpt=ckpt)

    def pending_interrupt(self) -> PendingInterrupt | None:
        """Return the primary active `PendingInterrupt` if the orchestration is.

        currently waiting on one, else `None`.
        """
        state = self._sdk.get_orchestration_state(self.instance_id, fetch_payloads=True)
        if (
            state is None
            or _status_name(state) in _TERMINAL
            or not state.serialized_custom_status
        ):
            return None
        status_obj = json.loads(state.serialized_custom_status) or {}
        pending = status_obj.get("pending")
        if not pending:
            return None
        return self._resolve_interrupt(
            pending, step=status_obj.get("step"), ckpt=status_obj.get("ckpt")
        )

    def pending_interrupts(self) -> list[PendingInterrupt]:
        """Return all currently pending `PendingInterrupt` receipts (including parallel.

        branch interrupts).
        """
        state = self._sdk.get_orchestration_state(self.instance_id, fetch_payloads=True)
        if (
            state is None
            or _status_name(state) in _TERMINAL
            or not state.serialized_custom_status
        ):
            return []
        status_obj = json.loads(state.serialized_custom_status) or {}
        items = status_obj.get("interrupts")
        if not items:
            pending = status_obj.get("pending")
            items = [pending] if pending else []
        step, ckpt = status_obj.get("step"), status_obj.get("ckpt")
        return [self._resolve_interrupt(entry, step=step, ckpt=ckpt) for entry in items]

    def _build_event_data(
        self,
        val: Any,
        *,
        ckpt: str | None = None,
        step: int | None = None,
        version: str | None = None,
        as_node: str | None = None,
        values: Any = _MISSING,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"value": pack(val)}
        if ckpt is not None:
            data["ckpt"] = str(ckpt)
        if step is not None:
            data["step"] = int(step)
        if version is not None:
            data["version"] = str(version)
        if as_node is not None:
            data["as_node"] = str(as_node)
        if values is not _MISSING:
            data["migrate_values"] = pack(values)
        return data

    def resume(
        self,
        receipt: PendingInterrupt
        | Command[Any]
        | Mapping[PendingInterrupt | str, Any]
        | str
        | Any = _MISSING,
        value: Any = _MISSING,
        *,
        version: str | None = None,
        as_node: str | None = None,
        values: Any = _MISSING,
    ) -> None:
        """Resume one or more pending interrupts by raising `"resume:<interrupt_id>"`.

        external events.

        Supports:
          - `run.resume(value)` (single positional argument) to resume the active
            pending interrupt(s) with `value`.
          - `run.resume(receipt, value)` or `run.resume(interrupt_id, value)` for a
            specific interrupt.
          - `run.resume({receipt_1: val_1, receipt_2: val_2})` for parallel interrupts.
          - `run.resume(Command(resume=...))` for native LangGraph `Command` style.
          - `run.resume()` (no args) to resume a static breakpoint (`interrupt_before` /
            `interrupt_after`).
          - `version`, `as_node`, `values`: optionally migrate the run to a new graph
            `version` (with optional
            `as_node` topology remap or state `values` update) as part of resuming.
        """
        targets = self.pending_interrupts()
        if not targets:
            single = self.pending_interrupt()
            targets = [single] if single is not None else []
        pending_by_id = {p.interrupt_id: p for p in targets}
        pending_ids = set(pending_by_id.keys())

        if receipt is _MISSING or receipt is None:
            if not targets:
                raise RunError("no pending interrupt to resume")
            if len(targets) > 1:
                raise ValueError(
                    "multiple interrupts are pending; pass a PendingInterrupt, "
                    "interrupt_id, or {interrupt_id: value} mapping"
                )
            actual_val = None if value is _MISSING else value
            t0 = targets[0]
            self._sdk.raise_orchestration_event(
                self.instance_id,
                RESUME_EVENT_PREFIX + t0.interrupt_id,
                data=self._build_event_data(
                    actual_val,
                    ckpt=t0.ckpt,
                    step=t0.step,
                    version=version,
                    as_node=as_node,
                    values=values,
                ),
            )
            return

        if isinstance(receipt, Command):
            if not targets:
                raise RunError("no pending interrupt to resume")
            res = receipt.resume
            if (
                isinstance(res, Mapping)
                and bool(res)
                and all(
                    isinstance(k, PendingInterrupt) or str(k) in pending_ids
                    for k in res
                )
            ):
                for k in res:
                    iid = k.interrupt_id if isinstance(k, PendingInterrupt) else str(k)
                    if iid not in pending_ids:
                        raise ValueError(f"unknown interrupt_id {iid!r}")
                for k, v in res.items():
                    iid = k.interrupt_id if isinstance(k, PendingInterrupt) else str(k)
                    matched = pending_by_id.get(iid)
                    if (
                        isinstance(k, PendingInterrupt)
                        and k.ckpt is not None
                        and matched is not None
                        and matched.ckpt is not None
                        and k.ckpt != matched.ckpt
                    ):
                        continue
                    eff_ckpt = (
                        k.ckpt
                        if isinstance(k, PendingInterrupt) and k.ckpt is not None
                        else (matched.ckpt if matched is not None else None)
                    )
                    eff_step = (
                        k.step
                        if isinstance(k, PendingInterrupt) and k.step is not None
                        else (matched.step if matched is not None else None)
                    )
                    self._sdk.raise_orchestration_event(
                        self.instance_id,
                        RESUME_EVENT_PREFIX + iid,
                        data=self._build_event_data(
                            v,
                            ckpt=eff_ckpt,
                            step=eff_step,
                            version=version,
                            as_node=as_node,
                            values=values,
                        ),
                    )
                return
            if len(targets) > 1:
                raise ValueError(
                    "multiple interrupts are pending; pass a PendingInterrupt, "
                    "interrupt_id, or {interrupt_id: value} mapping "
                    "whose keys match pending interrupt_ids"
                )
            t0 = targets[0]
            self._sdk.raise_orchestration_event(
                self.instance_id,
                RESUME_EVENT_PREFIX + t0.interrupt_id,
                data=self._build_event_data(
                    res,
                    ckpt=t0.ckpt,
                    step=t0.step,
                    version=version,
                    as_node=as_node,
                    values=values,
                ),
            )
            return

        if isinstance(receipt, PendingInterrupt):
            if pending_ids and receipt.interrupt_id not in pending_ids:
                raise ValueError(f"unknown interrupt_id {receipt.interrupt_id!r}")
            matched = pending_by_id.get(receipt.interrupt_id)
            if (
                receipt.ckpt is not None
                and matched is not None
                and matched.ckpt is not None
                and receipt.ckpt != matched.ckpt
            ):
                return
            eff_ckpt = (
                receipt.ckpt
                if receipt.ckpt is not None
                else (matched.ckpt if matched is not None else None)
            )
            eff_step = (
                receipt.step
                if receipt.step is not None
                else (matched.step if matched is not None else None)
            )
            self._sdk.raise_orchestration_event(
                self.instance_id,
                RESUME_EVENT_PREFIX + receipt.interrupt_id,
                data=self._build_event_data(
                    None if value is _MISSING else value,
                    ckpt=eff_ckpt,
                    step=eff_step,
                    version=version,
                    as_node=as_node,
                    values=values,
                ),
            )
            return

        if value is _MISSING:
            if not targets:
                raise RunError("no pending interrupt to resume")
            if (
                isinstance(receipt, Mapping)
                and bool(receipt)
                and all(
                    isinstance(k, PendingInterrupt) or str(k) in pending_ids
                    for k in receipt
                )
            ):
                for k in receipt:
                    iid = k.interrupt_id if isinstance(k, PendingInterrupt) else str(k)
                    if iid not in pending_ids:
                        raise ValueError(f"unknown interrupt_id {iid!r}")
                for k, v in receipt.items():
                    iid = k.interrupt_id if isinstance(k, PendingInterrupt) else str(k)
                    matched = pending_by_id.get(iid)
                    if (
                        isinstance(k, PendingInterrupt)
                        and k.ckpt is not None
                        and matched is not None
                        and matched.ckpt is not None
                        and k.ckpt != matched.ckpt
                    ):
                        continue
                    eff_ckpt = (
                        k.ckpt
                        if isinstance(k, PendingInterrupt) and k.ckpt is not None
                        else (matched.ckpt if matched is not None else None)
                    )
                    eff_step = (
                        k.step
                        if isinstance(k, PendingInterrupt) and k.step is not None
                        else (matched.step if matched is not None else None)
                    )
                    self._sdk.raise_orchestration_event(
                        self.instance_id,
                        RESUME_EVENT_PREFIX + iid,
                        data=self._build_event_data(
                            v,
                            ckpt=eff_ckpt,
                            step=eff_step,
                            version=version,
                            as_node=as_node,
                            values=values,
                        ),
                    )
                return
            if isinstance(receipt, str) and (
                receipt in pending_ids or receipt.startswith(STATIC_INTERRUPT_PREFIX)
            ):
                if pending_ids and receipt not in pending_ids:
                    raise ValueError(f"unknown interrupt_id {receipt!r}")
                matched = pending_by_id.get(receipt)
                self._sdk.raise_orchestration_event(
                    self.instance_id,
                    RESUME_EVENT_PREFIX + receipt,
                    data=self._build_event_data(
                        None,
                        ckpt=matched.ckpt if matched is not None else None,
                        step=matched.step if matched is not None else None,
                        version=version,
                        as_node=as_node,
                        values=values,
                    ),
                )
                return
            if len(targets) > 1:
                raise ValueError(
                    "multiple interrupts are pending; pass a PendingInterrupt, "
                    "interrupt_id, or {interrupt_id: value} mapping "
                    "whose keys match pending interrupt_ids"
                )
            t0 = targets[0]
            self._sdk.raise_orchestration_event(
                self.instance_id,
                RESUME_EVENT_PREFIX + t0.interrupt_id,
                data=self._build_event_data(
                    receipt,
                    ckpt=t0.ckpt,
                    step=t0.step,
                    version=version,
                    as_node=as_node,
                    values=values,
                ),
            )
            return

        interrupt_id = str(receipt)
        if pending_ids and interrupt_id not in pending_ids:
            raise ValueError(f"unknown interrupt_id {interrupt_id!r}")
        if not pending_ids and not interrupt_id.startswith(STATIC_INTERRUPT_PREFIX):
            raise RunError("no pending interrupt to resume")
        matched = pending_by_id.get(interrupt_id)
        self._sdk.raise_orchestration_event(
            self.instance_id,
            RESUME_EVENT_PREFIX + interrupt_id,
            data=self._build_event_data(
                value,
                ckpt=matched.ckpt if matched is not None else None,
                step=matched.step if matched is not None else None,
                version=version,
                as_node=as_node,
                values=values,
            ),
        )

    def upgrade(
        self,
        version: str,
        *,
        values: Any = _MISSING,
        as_node: str | None = None,
        resume: Any = _MISSING,
        receipt: PendingInterrupt | str | None = None,
    ) -> None:
        """Upgrade a paused run to `version` (optionally updating state `values`.

        remapping `as_node`, and passing a `resume` value).
        """
        self.resume(receipt, resume, version=version, as_node=as_node, values=values)

    def wait_pending(
        self, previous: PendingInterrupt | None = None, timeout: float = 60.0
    ) -> PendingInterrupt | None:
        """Poll until a new interrupt is published or the orchestration completes.

        Returns:
            A receipt distinct from previous, or None on completion without an
            interrupt.

        Raises:
            RunError: If the orchestration fails or terminates.
            TimeoutError: If no new interrupt arrives before the timeout.
        """
        deadline = time.monotonic() + timeout
        delay = 0.02
        while time.monotonic() < deadline:
            receipt = self.pending_interrupt()
            if receipt is not None:
                if previous is None or (
                    receipt.interrupt_id != previous.interrupt_id
                    or (
                        receipt.step is not None
                        and previous.step is not None
                        and receipt.step != previous.step
                    )
                    or (
                        receipt.ckpt is not None
                        and previous.ckpt is not None
                        and receipt.ckpt != previous.ckpt
                    )
                ):
                    return receipt
            st = self.status()
            if st in ("FAILED", "TERMINATED"):
                self.wait_result(timeout=1)
            if st == "COMPLETED":
                return None
            time.sleep(delay)
            delay = min(0.2, delay * 1.5)
        raise TimeoutError("no pending interrupt observed")

    def wait_result(self, timeout: float = 60.0) -> Any:
        """Block until the orchestration completes and return the final projected.

        LangGraph state.
        """
        state = self._sdk.wait_for_orchestration_completion(
            self.instance_id, timeout=max(1, math.ceil(timeout)), fetch_payloads=True
        )
        if state is None or state.runtime_status.name != "COMPLETED":
            detail = (
                state.failure_details.message
                if state is not None and state.failure_details
                else "no terminal state"
            )
            raise RunError(detail)
        output = json.loads(state.serialized_output)
        self.generations, self.steps = output["gen"] + 1, output["steps"]
        self.version = output.get("version")
        target_g = _resolve_graph(self._reg.graph, self.version)
        if "checkpoint_bundle" in output:
            last_cid = output.get("last_checkpoint_id") or output.get("ckpt")
            _hydrate_checkpoint_bundle(
                target_g,
                self.thread_id,
                output["checkpoint_bundle"],
                only_if_missing_cfg=_config(self.thread_id, last_cid)
                if last_cid
                else None,
                preserve_newer=True,
            )
        if output.get("output") is not None:
            return unpack(output["output"])
        if "result" in output:
            return unpack(output["result"])
        return read_output(target_g, self.thread_id, output.get("ckpt"))


@beta(message=_BETA_MESSAGE)
class AsyncRunClient:
    """Async client facade for starting and inspecting durable LangGraph runs on an async DTS client."""  # noqa: E501

    def __init__(self, registration: Registration, sdk: Any) -> None:
        self.registration = registration
        self.sdk = sdk

    async def start(
        self,
        input: Any,
        *,
        thread_id: str | None = None,
        instance_id: str | None = None,
        reuse_existing: bool = True,
        max_concurrency: int | None = None,
        node_max_concurrency: int | None = None,
        config: RunnableConfig | Mapping[str, Any] | None = None,
        context: Any = None,
        version: str | None = None,
        owner: str | None = None,
    ) -> AsyncRun:
        """Start or idempotently attach to a durable LangGraph run asynchronously."""
        reg = self.registration
        cfg_dict = dict(config) if isinstance(config, Mapping) else {}
        user_conf = (
            dict(cfg_dict.get("configurable") or {})
            if isinstance(cfg_dict.get("configurable"), Mapping)
            else {}
        )
        if max_concurrency is None and cfg_dict.get("max_concurrency") is not None:
            max_concurrency = int(cfg_dict["max_concurrency"])
        eff_mc = (
            reg.tool_config.max_concurrency
            if max_concurrency is None
            else int(max_concurrency)
        )
        if eff_mc is not None and eff_mc < 1:
            raise ValueError("max_concurrency must be >= 1 or None")
        eff_nmc = (
            reg.node_config.max_concurrency
            if node_max_concurrency is None
            else int(node_max_concurrency)
        )
        if eff_nmc is not None and eff_nmc < 1:
            raise ValueError("node_max_concurrency must be >= 1 or None")
        eff_ver = str(version) if version is not None else reg.version
        target_graph = _resolve_graph(reg.graph, eff_ver)
        explicit_thread = thread_id or (
            str(user_conf["thread_id"]) if user_conf.get("thread_id") else None
        )
        if (
            instance_id is not None
            and reuse_existing
            and hasattr(self.sdk, "get_orchestration_state")
        ):
            existing = await self.sdk.get_orchestration_state(
                instance_id, fetch_payloads=False
            )
            status_obj = (
                getattr(existing, "runtime_status", None)
                if existing is not None
                else None
            )
            status_name = getattr(status_obj, "name", status_obj)
            if status_name in _REUSABLE_STATUSES:
                if not getattr(existing, "serialized_input", None):
                    existing = await self.sdk.get_orchestration_state(
                        instance_id, fetch_payloads=True
                    )
                existing_input = _existing_run_input(existing, owner)
                thread = existing_input.get("thread") or explicit_thread or instance_id
                _check_checkpoint_owner(
                    await _aget_state(target_graph, _config(thread)), owner
                )
                return AsyncRun(reg, self.sdk, instance_id, thread)
        thread = explicit_thread or _default_thread(instance_id, owner)
        norm_cfg = _normalize_run_config(cfg_dict)
        snapshot = await _aget_state(target_graph, _config(thread, run_config=norm_cfg))
        _check_checkpoint_owner(snapshot, owner)
        expect = _checkpoint_id(snapshot)
        if reg.inline_checkpoints and expect is None and input is None:
            cached = _get_cached_inline_bundle(target_graph, thread)
            if cached is not None:
                _hydrate_checkpoint_bundle(target_graph, thread, cached)
                snapshot = await _aget_state(
                    target_graph, _config(thread, run_config=norm_cfg)
                )
                _check_checkpoint_owner(snapshot, owner)
                expect = _checkpoint_id(snapshot)
        if input is None and expect is None:
            raise ValueError("cannot start an empty thread without checkpoint state")
        request: dict[str, Any] = {
            "thread": thread,
            "kind": StepKind.START,
            "expect": expect,
            "input": pack(input),
        }
        iid = instance_id or _default_instance_id(
            reg.name, thread, expect, request["input"]
        )
        request["instance_id"] = iid
        if owner is not None:
            request["owner"] = owner
        if eff_ver is not None:
            request["version"] = eff_ver
        if eff_mc is not None:
            request["max_concurrency"] = eff_mc
        if eff_nmc is not None:
            request["node_max_concurrency"] = eff_nmc
        if norm_cfg is not None:
            request["run_config"] = pack(norm_cfg)
        if context is not None:
            request["context"] = _pack_context(context)
        if reg.inline_checkpoints and expect is not None:
            bundle = _export_checkpoint_bundle(target_graph, thread)
            if bundle is not None:
                request["checkpoint_bundle"] = bundle
        payload: dict[str, Any] = {
            "proto": 1,
            "thread": thread,
            "gen": 0,
            "steps": 0,
            "req": request,
            "step_name": reg.step_name,
            "tool_name": reg.tool_name,
            "node_name": reg.node_name,
            "limits": {
                "custom_status_limit": reg.limits.custom_status_limit,
                "max_steps": reg.limits.max_steps,
                "steps_per_generation": reg.limits.steps_per_generation,
                "byte_budget": reg.limits.byte_budget,
                "retries": reg.limits.retries,
                "tool_retries": reg.limits.tool_retries,
                "node_retries": reg.limits.node_retries,
                "async_timeout_seconds": reg.limits.async_timeout_seconds,
            },
        }
        if eff_ver is not None:
            payload["version"] = eff_ver
        if reg.upgrade_on_continue_as_new:
            payload["upgrade_on_can"] = True
            if reg.version is not None:
                payload["target_version"] = reg.version
        if eff_mc is not None:
            payload["max_concurrency"] = eff_mc
        if eff_nmc is not None:
            payload["node_max_concurrency"] = eff_nmc
        if "run_config" in request:
            payload["run_config"] = request["run_config"]
        if "context" in request:
            payload["context"] = request["context"]
        if owner is not None:
            payload["owner"] = str(owner)
        _check_payload(payload, "orchestrator input")
        await self.sdk.schedule_new_orchestration(
            reg.orchestrator_name, input=payload, instance_id=iid
        )
        return AsyncRun(reg, self.sdk, iid, thread)


@beta(message=_BETA_MESSAGE)
class AsyncRun:
    """Async handle to an in-flight or completed durable LangGraph orchestration."""

    def __init__(
        self, registration: Registration, sdk: Any, instance_id: str, thread_id: str
    ) -> None:
        self._reg = registration
        self._sdk = sdk
        self.instance_id = instance_id
        self.thread_id = thread_id
        self.generations: int | None = None
        self.steps: int | None = None
        self.version: str | None = None

    async def status(self) -> str:
        """Return the current DTS orchestration runtime status asynchronously."""
        state = await self._sdk.get_orchestration_state(
            self.instance_id, fetch_payloads=False
        )
        return state.runtime_status.name if state is not None else "UNKNOWN"

    async def pending_interrupts(self) -> list[PendingInterrupt]:
        """Return all currently pending interrupts asynchronously."""
        state = await self._sdk.get_orchestration_state(
            self.instance_id, fetch_payloads=True
        )
        if (
            state is None
            or _status_name(state) in _TERMINAL
            or not state.serialized_custom_status
        ):
            return []
        cs = json.loads(state.serialized_custom_status)
        if not cs:
            return []
        raw_items = cs.get("interrupts") or (
            [cs["pending"]] if cs.get("pending") else []
        )
        out: list[PendingInterrupt] = []
        for p in raw_items:
            iid = p["id"]
            val = unpack(p["value"]) if "value" in p else None
            out.append(
                PendingInterrupt(
                    interrupt_id=iid,
                    value=val,
                    step=cs.get("step"),
                    ckpt=cs.get("ckpt"),
                )
            )
        return out

    async def pending_interrupt(self) -> PendingInterrupt | None:
        """Return the primary pending interrupt asynchronously, or `None`."""
        items = await self.pending_interrupts()
        return items[0] if items else None

    async def wait_pending(
        self, previous: PendingInterrupt | None = None, timeout: float = 60.0
    ) -> PendingInterrupt | None:
        """Asynchronously poll until a new interrupt is published or the orchestration completes."""  # noqa: E501
        import asyncio as _aio

        deadline = time.monotonic() + timeout
        delay = 0.02
        while time.monotonic() < deadline:
            receipt = await self.pending_interrupt()
            if receipt is not None:
                if previous is None or (
                    receipt.interrupt_id != previous.interrupt_id
                    or (
                        receipt.step is not None
                        and previous.step is not None
                        and receipt.step != previous.step
                    )
                    or (
                        receipt.ckpt is not None
                        and previous.ckpt is not None
                        and receipt.ckpt != previous.ckpt
                    )
                ):
                    return receipt
            st = await self.status()
            if st in ("FAILED", "TERMINATED"):
                await self.wait_result(timeout=1)
            if st == "COMPLETED":
                return None
            await _aio.sleep(delay)
            delay = min(0.2, delay * 1.5)
        raise TimeoutError("no pending interrupt observed")

    async def resume(
        self,
        receipt: PendingInterrupt
        | str
        | Command[Any]
        | Mapping[str | PendingInterrupt, Any]
        | object
        | None = _MISSING,
        value: Any = _MISSING,
        *,
        version: str | None = None,
        as_node: str | None = None,
        values: Any = _MISSING,
    ) -> None:
        """Deliver a resume event asynchronously to the paused orchestration."""
        targets = await self.pending_interrupts()
        if isinstance(receipt, PendingInterrupt):
            actual_val = None if value is _MISSING else value
            payload: dict[str, Any] = {"value": pack(actual_val)}
            if receipt.ckpt is not None:
                payload["ckpt"] = receipt.ckpt
            if receipt.step is not None:
                payload["step"] = receipt.step
            if version is not None:
                payload["version"] = str(version)
            if as_node is not None:
                payload["as_node"] = str(as_node)
            if values is not _MISSING:
                payload["migrate_values"] = pack(values)
            await self._sdk.raise_orchestration_event(
                self.instance_id,
                RESUME_EVENT_PREFIX + receipt.interrupt_id,
                data=payload,
            )
            return
        if not targets:
            raise RunError("no pending interrupt to resume")
        t0 = targets[0]
        actual_val = (
            None
            if (receipt is _MISSING and value is _MISSING)
            else (receipt if value is _MISSING else value)
        )
        payload = {"value": pack(actual_val)}
        if t0.ckpt is not None:
            payload["ckpt"] = t0.ckpt
        if t0.step is not None:
            payload["step"] = t0.step
        await self._sdk.raise_orchestration_event(
            self.instance_id,
            RESUME_EVENT_PREFIX + t0.interrupt_id,
            data=payload,
        )

    async def wait_result(self, timeout: float = 60.0) -> Any:
        """Await orchestration completion and return the final projected LangGraph state."""  # noqa: E501
        state = await self._sdk.wait_for_orchestration_completion(
            self.instance_id, timeout=max(1, math.ceil(timeout)), fetch_payloads=True
        )
        if state is None or state.runtime_status.name != "COMPLETED":
            detail = (
                state.failure_details.message
                if state is not None and state.failure_details
                else "no terminal state"
            )
            raise RunError(detail)
        output = json.loads(state.serialized_output)
        self.generations, self.steps = output["gen"] + 1, output["steps"]
        self.version = output.get("version")
        target_g = _resolve_graph(self._reg.graph, self.version)
        if "checkpoint_bundle" in output:
            last_cid = output.get("last_checkpoint_id") or output.get("ckpt")
            _hydrate_checkpoint_bundle(
                target_g,
                self.thread_id,
                output["checkpoint_bundle"],
                only_if_missing_cfg=_config(self.thread_id, last_cid)
                if last_cid
                else None,
                preserve_newer=True,
            )
        if output.get("output") is not None:
            return unpack(output["output"])
        if "result" in output:
            return unpack(output["result"])
        snapshot = await _aget_state(
            target_g, _config(self.thread_id, output.get("ckpt"))
        )
        return _project_output(target_g, snapshot.values)
