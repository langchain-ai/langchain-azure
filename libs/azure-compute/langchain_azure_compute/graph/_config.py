# Copyright (c) Microsoft. All rights reserved.

"""Configuration, limits, tool durability policies, and value serialization.

This module defines the static configuration and serialization primitives shared
across the client (`RunClient`), worker registration (`Registration`), and
execution engine (`runner.py`):

1. **Execution Limits (`Limits`):** Caps superstep count, history size per
   orchestration generation (`ContinueAsNew`), custom-status payload bytes, and
   activity retry attempts.
2. **Typed Serialization (`pack` / `unpack`):** Wraps LangGraph's
   `JsonPlusSerializer` (`msgpack` / typed JSON) into base64-encoded JSON
   envelopes (`{"t": ..., "b64": ...}`) so arbitrary LangChain messages,
   `Command` objects, and custom types cross Durable Task JSON boundaries intact.
3. **Tool Durability Policy (`ToolConfig`, `@durable_tool`, `mark_tool`):**
   Resolves whether each tool inside a LangGraph `ToolNode` executes as its own
   durable DTS activity (with independent retries and `interrupt()` support) or
   runs locally inside the superstep activity.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import (
    Any,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    overload,
    runtime_checkable,
)

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.prebuilt import ToolNode
from langgraph.pregel import Pregel

_SERDE = JsonPlusSerializer()


class ExecutionMode(StrEnum):
    """Execution mode for tools and nodes (`'durable'` vs `'local'`)."""

    DURABLE = "durable"
    LOCAL = "local"


class StepKind(StrEnum):
    """Command discriminator (`request['kind']`) for the `run_step` activity."""

    START = "start"
    STEP = "step"
    RESUME = "resume"
    TOOLS = "tools"
    NODES = "nodes"
    MIGRATE = "migrate"
    INSPECT = "inspect"


class StepStatus(StrEnum):
    """Status discriminator returned by `run_step`, `run_tool`, `run_node`, and.

    `make_orchestrator`.
    """

    RUNNING = "running"
    DONE = "done"
    INTERRUPTED = "interrupted"
    TOOLS = "tools"
    NODES = "nodes"
    OK = "ok"


class TaskResultStatus(StrEnum):
    """Status discriminator returned by `run_tool` and `run_node`."""

    OK = "ok"
    INTERRUPTED = "interrupted"


RESUME_EVENT_PREFIX = "resume:"
STATIC_INTERRUPT_PREFIX = "static:"
STATIC_BEFORE_PREFIX = "static:before:"
STATIC_AFTER_PREFIX = "static:after:"
DURABLE_TOOL_ERROR_KEY = "__durable_tool_error__"

# Public & internal type aliases
CompiledGraph: TypeAlias = Pregel[Any, Any, Any, Any]
Checkpointer: TypeAlias = BaseCheckpointSaver[Any]
PolicyValue: TypeAlias = bool | int | ExecutionMode | str
ToolCallable: TypeAlias = BaseTool | Callable[..., Any]
NodeCallable: TypeAlias = Runnable[Any, Any] | Callable[..., Any]
Selector: TypeAlias = str | ToolCallable | NodeCallable | tuple[str, str]
MigrationKey: TypeAlias = tuple[str | None, str | None] | str
MigrationSpec: TypeAlias = Callable[..., Any] | Mapping[str, Any]

_F = TypeVar("_F", bound=Callable[..., Any] | BaseTool | Runnable[Any, Any])


class PackedBlob(TypedDict):
    """Serialized value envelope produced by `pack()` and consumed by `unpack()`."""

    t: str
    b64: str


class CheckpointBundle(TypedDict, total=False):
    """Exported `InMemorySaver` checkpoint slice carried across DTS activities in.

    Zero-Config mode.
    """

    v: int
    storage: dict[str, dict[str, list[Any]]]
    writes: list[list[Any]]
    blobs: list[list[Any]]


class InterruptEntry(TypedDict, total=False):
    """Serialized interrupt descriptor inside `StepReport`, `ToolReport`, `NodeReport`.

    and custom status.
    """

    id: str
    value: PackedBlob
    in_checkpoint: bool
    static: bool


class ToolCallSpec(TypedDict, total=False):
    """Per-tool execution descriptor emitted by `_report()` when pausing before a.

    `ToolNode`.
    """

    call_id: str
    index: int
    id: str | None
    name: str
    call: PackedBlob
    durable: bool
    retries: int
    handle_errors: bool


class NodeTaskSpec(TypedDict, total=False):
    """Per-node execution descriptor emitted by `_report()` when pausing before.

    `@durable_node` tasks.
    """

    call_id: str
    index: int
    task_id: str
    name: str
    input: PackedBlob
    durable: bool
    retries: int


class StepRequest(TypedDict, total=False):
    """Input payload for the `langgraph.<name>.step` activity (`run_step`)."""

    thread: str
    kind: StepKind | str
    expect: str | None
    input: PackedBlob
    resume: PackedBlob
    node: str
    tool_ns: str
    outputs: list[PackedBlob] | Mapping[str, PackedBlob] | Mapping[int, PackedBlob]
    version: str | None
    migrate_from: str | None
    as_node: str | None
    migrate_values: PackedBlob
    max_concurrency: int | None
    node_max_concurrency: int | None
    run_config: PackedBlob
    config: Mapping[str, Any] | None
    context: PackedBlob
    checkpoint_bundle: CheckpointBundle | Mapping[str, Any]


class StepReport(TypedDict, total=False):
    """Control envelope returned by `run_step` / `_report`."""

    status: StepStatus | str
    ckpt: str | None
    replayed: bool
    jumped: bool
    target_version: str
    checkpoint_bundle: CheckpointBundle
    interrupt: InterruptEntry
    interrupts: list[InterruptEntry]
    output: PackedBlob
    result: PackedBlob
    node: str
    tool_calls: list[ToolCallSpec]
    tool_ns: str
    tool_ckpt: str
    node_tasks: list[NodeTaskSpec]
    max_concurrency: int


class ToolRequest(TypedDict, total=False):
    """Input payload for the `langgraph.<name>.tool` activity (`run_tool`)."""

    thread: str
    ckpt: str | None
    node: str
    call_id: str
    index: int
    call: PackedBlob
    version: str | None
    tool_ns: str
    tool_ckpt: str
    resumes: list[PackedBlob]
    run_config: PackedBlob
    config: Mapping[str, Any] | None
    context: PackedBlob
    checkpoint_bundle: CheckpointBundle | Mapping[str, Any]


class ToolReport(TypedDict, total=False):
    """Result envelope returned by `run_tool`."""

    call_id: str
    status: TaskResultStatus | StepStatus | str
    output: PackedBlob
    interrupt: InterruptEntry


class NodeRequest(TypedDict, total=False):
    """Input payload for the `langgraph.<name>.node` activity (`run_node`)."""

    thread: str
    ckpt: str | None
    call_id: str
    index: int
    task_id: str
    name: str
    input: PackedBlob | None
    version: str | None
    resumes: list[PackedBlob]
    run_config: PackedBlob
    config: Mapping[str, Any] | None
    context: PackedBlob
    checkpoint_bundle: CheckpointBundle | Mapping[str, Any]


class NodeReport(TypedDict, total=False):
    """Result envelope returned by `run_node`."""

    call_id: str
    status: TaskResultStatus | StepStatus | str
    output: PackedBlob
    interrupt: InterruptEntry


class OrchestratorInput(TypedDict, total=False):
    """Input payload for the `langgraph.<name>.run` orchestrator."""

    proto: int
    thread: str
    gen: int
    steps: int
    req: StepRequest | dict[str, Any]
    step_name: str
    tool_name: str
    node_name: str
    version: str | None
    target_version: str | None
    upgrade_on_can: bool | None
    max_concurrency: int | None
    node_max_concurrency: int | None
    run_config: PackedBlob
    context: PackedBlob
    checkpoint_bundle: CheckpointBundle | Mapping[str, Any]


class OrchestratorOutput(TypedDict, total=False):
    """Terminal output payload returned by the `langgraph.<name>.run` orchestrator."""

    status: StepStatus | str
    thread: str
    steps: int
    gen: int
    ckpt: str | None
    version: str | None
    result: PackedBlob
    output: PackedBlob
    checkpoint_bundle: CheckpointBundle | Mapping[str, Any]


@runtime_checkable
class WorkerProtocol(Protocol):
    """Structural protocol for a Durable Task worker (`DurableTaskSchedulerWorker` or.

    `TaskHubGrpcWorker`).
    """

    def add_orchestrator(self, fn: Callable[..., Any]) -> Any: ...
    def add_activity(self, fn: Callable[..., Any]) -> Any: ...


@runtime_checkable
class ClientProtocol(Protocol):
    """Structural protocol for a Durable Task client (`DurableTaskSchedulerClient` or.

    `TaskHubGrpcClient`).
    """

    def schedule_new_orchestration(
        self, orchestrator: Any, *, input: Any = ..., instance_id: str | None = ...
    ) -> Any: ...
    def get_orchestration_state(
        self, instance_id: str, *, fetch_payloads: bool = ...
    ) -> Any: ...
    def raise_orchestration_event(
        self, instance_id: str, event_name: str, *, data: Any = ...
    ) -> Any: ...
    def wait_for_orchestration_completion(
        self, instance_id: str, *, timeout: int = ..., fetch_payloads: bool = ...
    ) -> Any: ...


# DTS custom status has a strict server cap (~16-64 KB); keep interrupt custom status
# well below it.
CUSTOM_STATUS_INLINE_LIMIT = 16 * 1024

# Maximum length of the user-provided `name` passed to `attach_durability(graph,
# name=...)`.
# Keeps the full `langgraph.<name>.step` / `langgraph.<name>.run` /
# `langgraph.<name>.tool`
# task identifier <= 79 characters, well within DTS, OpenTelemetry, and storage
# identifier limits.
MAX_GRAPH_NAME_LENGTH = 64
_GRAPH_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _validate_graph_name(name: str) -> str:
    """Validate that `name` is a human-readable identifier within.

    `MAX_GRAPH_NAME_LENGTH`.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("graph name must be a non-empty string")
    if len(name) > MAX_GRAPH_NAME_LENGTH:
        raise ValueError(
            f"graph name {name!r} is {len(name)} characters; "
            f"maximum allowed length is {MAX_GRAPH_NAME_LENGTH}"
        )
    if not _GRAPH_NAME_RE.fullmatch(name):
        raise ValueError(
            f"graph name {name!r} must start with an alphanumeric "
            "character and contain only ASCII letters, digits, "
            "'_', '-', or '.' (no whitespace or control characters)"
        )
    return name


class RunError(RuntimeError):
    """Raised when a durable LangGraph orchestration fails, is terminated, or violates.

    execution limits.
    """


@dataclass(frozen=True)
class Limits:
    """Execution and payload budgets for a durable LangGraph run.

    Attributes:
        custom_status_limit: Maximum serialized byte size for inlining an interrupt
        payload
            into the DTS orchestration custom status. Larger payloads stay in the
            checkpointer.
        max_steps: Hard ceiling on total graph supersteps before raising `RunError`.
        steps_per_generation: Maximum activity calls per orchestration generation before
            rolling over history via `continue_as_new`.
        byte_budget: Maximum cumulative control-plane payload bytes per generation
        before
            rolling over history via `continue_as_new`.
        retries: Default maximum attempts for superstep activities (and tool activities
        when
            `tool_retries` is not set).
        tool_retries: Optional default maximum attempts specifically for durable tool
        activities.
    """

    custom_status_limit: int = CUSTOM_STATUS_INLINE_LIMIT
    max_steps: int = 10_000
    steps_per_generation: int = 250
    byte_budget: int = 3_000_000
    retries: int = 3
    tool_retries: int | None = None
    node_retries: int | None = None


@dataclass
class OrchestratorState:
    """Strongly typed runtime state for `run_graph_orchestrator` across supersteps and.

    `continue_as_new`.
    """

    thread: str
    gen: int
    steps: int
    request: dict[str, Any]
    step_name: str
    tool_name: str
    node_name: str
    version: str | None = None
    target_version: str | None = None
    upgrade_on_can: bool = False
    max_concurrency: int | None = None
    node_max_concurrency: int | None = None
    run_config: PackedBlob | None = None
    context_blob: PackedBlob | None = None
    checkpoint_bundle: CheckpointBundle | None = None

    @classmethod
    def from_payload(
        cls, inp: OrchestratorInput | Mapping[str, Any]
    ) -> OrchestratorState:
        """Validate and normalize an `OrchestratorInput` dict into a strongly typed.

        `OrchestratorState`.
        """
        proto = inp.get("proto", 1)
        if proto != 1:
            raise RunError(
                f"unsupported orchestrator protocol version {proto!r} (expected 1)"
            )
        req: dict[str, Any] = dict(inp["req"])
        step_name = str(inp["step_name"])
        prefix = step_name.rsplit(".", 1)[0]
        raw_mc = inp.get("max_concurrency", req.get("max_concurrency"))
        raw_nmc = inp.get("node_max_concurrency", req.get("node_max_concurrency"))
        version = inp.get("version", req.get("version"))
        target_version = inp.get("target_version")
        run_config: Any = inp.get("run_config", req.get("run_config"))
        context_blob: Any = inp.get("context", req.get("context"))
        checkpoint_bundle: Any = inp.get("checkpoint_bundle") or req.get(
            "checkpoint_bundle"
        )
        state = cls(
            thread=str(inp["thread"]),
            gen=int(inp["gen"]),
            steps=int(inp["steps"]),
            request=req,
            step_name=step_name,
            tool_name=str(inp.get("tool_name") or f"{prefix}.tool"),
            node_name=str(inp.get("node_name") or f"{prefix}.node"),
            version=str(version) if version is not None else None,
            target_version=str(target_version) if target_version is not None else None,
            upgrade_on_can=bool(inp.get("upgrade_on_can", False)),
            max_concurrency=max(1, int(raw_mc)) if raw_mc is not None else None,
            node_max_concurrency=max(1, int(raw_nmc)) if raw_nmc is not None else None,
            run_config=run_config,
            context_blob=context_blob,
            checkpoint_bundle=checkpoint_bundle,
        )
        state.apply_step_options_to(state.request, preserve_existing=True)
        return state

    def apply_context_to(
        self, req: dict[str, Any], *, preserve_existing: bool = False
    ) -> dict[str, Any]:
        """Populate `version`, `run_config`, `context`, and `checkpoint_bundle` on an.

        activity request dict.
        """
        for key, val in (
            ("version", self.version),
            ("run_config", self.run_config),
            ("context", self.context_blob),
            ("checkpoint_bundle", self.checkpoint_bundle),
        ):
            if val is not None and (not preserve_existing or key not in req):
                req[key] = val
        return req

    def apply_step_options_to(
        self, req: dict[str, Any], *, preserve_existing: bool = False
    ) -> dict[str, Any]:
        """Populate active context and concurrency caps (`max_concurrency`.

        `node_max_concurrency`) on a step request.
        """
        self.apply_context_to(req, preserve_existing=preserve_existing)
        if self.max_concurrency is not None and (
            not preserve_existing or "max_concurrency" not in req
        ):
            req["max_concurrency"] = self.max_concurrency
        if self.node_max_concurrency is not None and (
            not preserve_existing or "node_max_concurrency" not in req
        ):
            req["node_max_concurrency"] = self.node_max_concurrency
        return req

    def to_can_payload(self, *, target_version: str | None = None) -> OrchestratorInput:
        """Construct the `continue_as_new` payload for the next orchestration.

        generation (`gen + 1`).
        """
        req = self.apply_step_options_to(dict(self.request))
        can_payload: OrchestratorInput = {
            "proto": 1,
            "thread": self.thread,
            "gen": self.gen + 1,
            "steps": self.steps,
            "req": req,
            "step_name": self.step_name,
            "tool_name": self.tool_name,
            "node_name": self.node_name,
        }
        eff_target = (
            target_version if target_version is not None else self.target_version
        )
        if self.version is not None:
            can_payload["version"] = self.version
        if eff_target is not None:
            can_payload["target_version"] = eff_target
        if self.upgrade_on_can:
            can_payload["upgrade_on_can"] = True
        if self.max_concurrency is not None:
            can_payload["max_concurrency"] = self.max_concurrency
        if self.node_max_concurrency is not None:
            can_payload["node_max_concurrency"] = self.node_max_concurrency
        if self.run_config is not None:
            can_payload["run_config"] = self.run_config
        if self.context_blob is not None:
            can_payload["context"] = self.context_blob
        if self.checkpoint_bundle is not None:
            can_payload["checkpoint_bundle"] = self.checkpoint_bundle
        return can_payload


def pack(value: Any) -> PackedBlob:
    """Serialize `value` with LangGraph's `JsonPlusSerializer` into a JSON-safe dict."""
    kind, data = _SERDE.dumps_typed(value)
    return {"t": kind, "b64": base64.b64encode(data).decode("ascii")}


def unpack(blob: PackedBlob | Mapping[str, Any]) -> Any:
    """Deserialize a `pack()`ed dict back into its native LangGraph/LangChain object."""
    return _SERDE.loads_typed((blob["t"], base64.b64decode(blob["b64"])))


def _size(value: Any) -> int:
    """Return the compact UTF-8 JSON byte length of `value` for budget accounting."""
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")))


@overload
def mark_tool(
    tool: _F,
    *,
    durable: bool = True,
    execution: ExecutionMode | str | None = None,
    retries: int | None = None,
) -> _F: ...


@overload
def mark_tool(
    tool: None = None,
    *,
    durable: bool = True,
    execution: ExecutionMode | str | None = None,
    retries: int | None = None,
) -> Callable[[_F], _F]: ...


def mark_tool(
    tool: _F | None = None,
    *,
    durable: bool = True,
    execution: ExecutionMode | str | None = None,
    retries: int | None = None,
) -> _F | Callable[[_F], _F]:
    """Mark a LangChain tool or callable as durable (or local) with an optional.

    per-tool retry count.
    """
    if tool is None:
        return lambda fn: mark_tool(
            fn, durable=durable, execution=execution, retries=retries
        )
    is_durable = (
        (execution == ExecutionMode.DURABLE) if execution is not None else bool(durable)
    )
    setattr(tool, "_durable_tool", is_durable)
    if retries is not None:
        setattr(tool, "_durable_retries", int(retries))
    return tool


@overload
def durable_tool(
    tool: _F, *, durable: bool = True, retries: int | None = None
) -> _F: ...


@overload
def durable_tool(
    tool: None = None, *, durable: bool = True, retries: int | None = None
) -> Callable[[_F], _F]: ...


def durable_tool(
    tool: _F | None = None,
    *,
    durable: bool = True,
    retries: int | None = None,
) -> _F | Callable[[_F], _F]:
    """Decorator to configure tool durability: `@durable_tool`.

    `@durable_tool(retries=4)`, or `@durable_tool(durable=False)`.
    """
    if tool is not None:
        return mark_tool(tool, durable=durable, retries=retries)
    return lambda fn: mark_tool(fn, durable=durable, retries=retries)


@overload
def mark_node(
    node: _F,
    *,
    durable: bool = True,
    execution: ExecutionMode | str | None = None,
    retries: int | None = None,
) -> _F: ...


@overload
def mark_node(
    node: None = None,
    *,
    durable: bool = True,
    execution: ExecutionMode | str | None = None,
    retries: int | None = None,
) -> Callable[[_F], _F]: ...


def mark_node(
    node: _F | None = None,
    *,
    durable: bool = True,
    execution: ExecutionMode | str | None = None,
    retries: int | None = None,
) -> _F | Callable[[_F], _F]:
    """Mark a LangGraph node callable or runnable as a durable activity (or local) with.

    an optional retry count.
    """
    if node is None:
        return lambda fn: mark_node(
            fn, durable=durable, execution=execution, retries=retries
        )
    is_durable = (
        (execution == ExecutionMode.DURABLE) if execution is not None else bool(durable)
    )
    setattr(node, "_durable_node", is_durable)
    if retries is not None:
        setattr(node, "_durable_retries", int(retries))
    return node


@overload
def durable_node(
    node: _F, *, durable: bool = True, retries: int | None = None
) -> _F: ...


@overload
def durable_node(
    node: None = None, *, durable: bool = True, retries: int | None = None
) -> Callable[[_F], _F]: ...


def durable_node(
    node: _F | None = None,
    *,
    durable: bool = True,
    retries: int | None = None,
) -> _F | Callable[[_F], _F]:
    """Decorator to configure node durability: `@durable_node`.

    `@durable_node(retries=4)`, or `@durable_node(durable=False)`.

    When a node is marked with `@durable_node` (or enabled via `attach_durability(...,
    durable_nodes=...)`),
    each invocation of that node - including parallel fan-out branches and dynamic
    `Send(node, arg)`
    map-reduce targets - executes as its own isolated `langgraph.<name>.node` Durable
    Task activity with
    independent retries and cross-worker distribution.
    """
    if node is not None:
        return mark_node(node, durable=durable, retries=retries)
    return lambda fn: mark_node(fn, durable=durable, retries=retries)


@dataclass
class ToolConfig:
    """Resolved durability, retry, and concurrency policy for tools inside LangGraph.

    `ToolNode`s.

    Precedence order (highest to lowest):
      1. Scoped node+tool key in `policy` / `retries` (e.g. `"subgraph:tools::search"`
         or `"tools::search"`)
      2. Tool name in `policy` / `retries` (e.g. `"search"`)
      3. Decorator attributes on the tool object (`@durable_tool(retries=...)` /
         `mark_tool(...)`)
      4. `ToolConfig.default_durable` / `ToolConfig.default_retries`
      5. `Limits.tool_retries` -> `Limits.retries`

    `max_concurrency` controls how many tool calls in a single `ToolNode` superstep run
    concurrently:
      - `None` (default): fan out all durable tool calls in parallel in a single batch.
      - `1`: execute tool calls strictly sequentially one at a time in order.
      - `N > 1`: execute tool calls in bounded concurrent batches of size `N`.
    """

    default_durable: bool = True
    policy: dict[str, bool] | None = None
    retries: dict[str, int] | None = None
    default_retries: int | None = None
    max_concurrency: int | None = None

    def __post_init__(self) -> None:
        if self.max_concurrency is not None:
            mc = int(self.max_concurrency)
            if mc < 1:
                raise ValueError("max_concurrency must be >= 1 or None")
            self.max_concurrency = mc

    @classmethod
    def from_options(
        cls,
        durable_tools: bool
        | ExecutionMode
        | str
        | bytes
        | Iterable[Selector]
        | Mapping[Selector, PolicyValue] = True,
        tool_policy: Mapping[Selector, PolicyValue] | None = None,
        tool_retries: int | Mapping[Selector, int] | None = None,
        max_concurrency: int | None = None,
        tool_max_concurrency: int | None = None,
    ) -> ToolConfig:
        """Normalize user-facing `attach_durability()` tool options into a.

        `ToolConfig`.
        """
        policy: dict[str, bool] = {}
        retries: dict[str, int] = {}
        default_durable = True
        default_retries: int | None = None
        eff_mc = (
            max_concurrency if max_concurrency is not None else tool_max_concurrency
        )

        if isinstance(durable_tools, bool):
            default_durable = durable_tools
        elif isinstance(durable_tools, (str, bytes)):
            is_dur, r = _parse_policy_value(
                durable_tools.decode("utf-8")
                if isinstance(durable_tools, bytes)
                else durable_tools
            )
            default_durable = is_dur
            if r is not None:
                default_retries = r
        elif isinstance(durable_tools, Mapping):
            default_durable = False
            for k, v in durable_tools.items():
                key = _selector_key(k)
                is_dur, r = _parse_policy_value(v)
                policy[key] = is_dur
                if r is not None:
                    retries[key] = r
        elif isinstance(durable_tools, Iterable):
            default_durable = False
            for item in durable_tools:
                name = getattr(item, "name", None) or _selector_key(item)
                policy[str(name)] = True
        else:
            raise ValueError(f"unsupported durable_tools option {durable_tools!r}")

        if tool_policy is not None:
            for k, v in tool_policy.items():
                key = _selector_key(k)
                is_dur, r = _parse_policy_value(v)
                policy[key] = is_dur
                if r is not None:
                    retries[key] = r

        if isinstance(tool_retries, int):
            default_retries = tool_retries
        elif isinstance(tool_retries, Mapping):
            for k, r_val in tool_retries.items():
                retries[_selector_key(k)] = int(r_val)

        return cls(
            default_durable=default_durable,
            policy=policy,
            retries=retries,
            default_retries=default_retries,
            max_concurrency=eff_mc,
        )

    def resolve(
        self, tool_node: ToolNode | Any, node_name: str, tool_name: str, limits: Limits
    ) -> tuple[bool, int]:
        """Return `(is_durable, max_attempts)` for `tool_name` on `node_name`."""
        tool_obj = getattr(tool_node, "tools_by_name", {}).get(tool_name)
        bare_node = node_name.rsplit(":", 1)[-1] if ":" in node_name else node_name
        scoped = f"{node_name}::{tool_name}"
        bare_scoped = f"{bare_node}::{tool_name}"
        policy = self.policy or {}
        if scoped in policy:
            is_durable = policy[scoped]
        elif bare_scoped in policy:
            is_durable = policy[bare_scoped]
        elif tool_name in policy:
            is_durable = policy[tool_name]
        elif tool_obj is not None and hasattr(tool_obj, "_durable_tool"):
            is_durable = bool(getattr(tool_obj, "_durable_tool"))
        else:
            is_durable = self.default_durable

        retries_map = self.retries or {}
        if scoped in retries_map:
            attempts = retries_map[scoped]
        elif bare_scoped in retries_map:
            attempts = retries_map[bare_scoped]
        elif tool_name in retries_map:
            attempts = retries_map[tool_name]
        elif tool_obj is not None and hasattr(tool_obj, "_durable_retries"):
            attempts = int(getattr(tool_obj, "_durable_retries"))
        elif self.default_retries is not None:
            attempts = self.default_retries
        elif limits.tool_retries is not None:
            attempts = limits.tool_retries
        else:
            attempts = limits.retries
        return is_durable, max(1, int(attempts))


def _effective_tool_config(
    graph: CompiledGraph, tool_config: ToolConfig | None = None
) -> ToolConfig:
    """Return the explicit `tool_config`, the graph's attached config, or a.

    conservative default.
    """
    if tool_config is not None:
        return tool_config
    cfg = getattr(graph, "_durable_tool_config", None)
    if isinstance(cfg, ToolConfig):
        return cfg
    # When run_step is called directly without attach_durability(), only explicitly
    # marked tools are durable.
    return ToolConfig(default_durable=False)


def _node_candidates(graph: CompiledGraph, node_name: str) -> list[Any]:
    """Return candidate objects carrying node durability attributes.

    Inspect node specs, runnable wrappers, and underlying sync and async callables.
    """
    out: list[Any] = []
    builder = getattr(graph, "builder", None)
    bnodes = getattr(builder, "nodes", None) or {}
    spec = bnodes.get(node_name)
    if spec is not None:
        out.append(spec)
        runnable = getattr(spec, "runnable", None)
        if runnable is not None:
            out.append(runnable)
            if getattr(runnable, "func", None) is not None:
                out.append(runnable.func)
            if getattr(runnable, "afunc", None) is not None:
                out.append(runnable.afunc)
    pnodes = getattr(graph, "nodes", None) or {}
    pnode = pnodes.get(node_name)
    if pnode is not None:
        out.append(pnode)
        bound = getattr(pnode, "bound", None)
        if bound is not None:
            out.append(bound)
            if getattr(bound, "func", None) is not None:
                out.append(bound.func)
            if getattr(bound, "afunc", None) is not None:
                out.append(bound.afunc)
    return out


def _is_tool_or_subgraph_node(graph: CompiledGraph, node_name: str) -> bool:
    """Return True if `node_name` is a `ToolNode` or compiled subgraph node."""
    builder = getattr(graph, "builder", None)
    bnodes = getattr(builder, "nodes", None) or {}
    spec = bnodes.get(node_name)
    runnable = getattr(spec, "runnable", spec)
    if (
        runnable is not None
        and hasattr(runnable, "tools_by_name")
        and hasattr(runnable, "_parse_input")
    ):
        return True
    pnodes = getattr(graph, "nodes", None) or {}
    pnode = pnodes.get(node_name)
    if pnode is not None and getattr(pnode, "subgraphs", None):
        return True
    return False


@dataclass
class NodeConfig:
    """Resolved durability, retry, and concurrency policy for LangGraph nodes.

    (including parallel fan-out and `Send()` targets).

    By default (`default_durable=False`), only nodes explicitly decorated with
    `@durable_node` / `mark_node`
    or listed in `durable_nodes` / `node_policy` run as separate `langgraph.<name>.node`
    activities.
    When `durable_nodes=True`, all regular graph nodes run as separate activities
    (similar to Temporal's per-node activity model).
    """

    default_durable: bool = False
    policy: dict[str, bool] | None = None
    retries: dict[str, int] | None = None
    default_retries: int | None = None
    max_concurrency: int | None = None

    def __post_init__(self) -> None:
        if self.max_concurrency is not None:
            mc = int(self.max_concurrency)
            if mc < 1:
                raise ValueError("node_max_concurrency must be >= 1 or None")
            self.max_concurrency = mc

    @classmethod
    def from_options(
        cls,
        durable_nodes: bool
        | ExecutionMode
        | str
        | bytes
        | Iterable[Selector]
        | Mapping[Selector, PolicyValue] = False,
        node_policy: Mapping[Selector, PolicyValue] | None = None,
        node_retries: int | Mapping[Selector, int] | None = None,
        node_max_concurrency: int | None = None,
    ) -> NodeConfig:
        """Normalize user-facing `attach_durability()` node options into a.

        `NodeConfig`.
        """
        policy: dict[str, bool] = {}
        retries: dict[str, int] = {}
        default_durable = False
        default_retries: int | None = None

        if isinstance(durable_nodes, bool):
            default_durable = durable_nodes
        elif isinstance(durable_nodes, (str, bytes)):
            is_dur, r = _parse_policy_value(
                durable_nodes.decode("utf-8")
                if isinstance(durable_nodes, bytes)
                else durable_nodes
            )
            default_durable = is_dur
            if r is not None:
                default_retries = r
        elif isinstance(durable_nodes, Mapping):
            default_durable = False
            for k, v in durable_nodes.items():
                key = (
                    getattr(k, "name", None)
                    or getattr(k, "__name__", None)
                    or _selector_key(k)
                )
                is_dur, r = _parse_policy_value(v)
                policy[str(key)] = is_dur
                if r is not None:
                    retries[str(key)] = r
        elif isinstance(durable_nodes, Iterable):
            default_durable = False
            for item in durable_nodes:
                name = (
                    getattr(item, "name", None)
                    or getattr(item, "__name__", None)
                    or _selector_key(item)
                )
                policy[str(name)] = True
        else:
            raise ValueError(f"unsupported durable_nodes option {durable_nodes!r}")

        if node_policy is not None:
            for k, v in node_policy.items():
                key = (
                    getattr(k, "name", None)
                    or getattr(k, "__name__", None)
                    or _selector_key(k)
                )
                is_dur, r = _parse_policy_value(v)
                policy[str(key)] = is_dur
                if r is not None:
                    retries[str(key)] = r

        if isinstance(node_retries, int):
            default_retries = node_retries
        elif isinstance(node_retries, Mapping):
            for k, r_val in node_retries.items():
                key = (
                    getattr(k, "name", None)
                    or getattr(k, "__name__", None)
                    or _selector_key(k)
                )
                retries[str(key)] = int(r_val)

        return cls(
            default_durable=default_durable,
            policy=policy,
            retries=retries,
            default_retries=default_retries,
            max_concurrency=node_max_concurrency,
        )

    def resolve(
        self, graph: CompiledGraph, node_name: str, limits: Limits
    ) -> tuple[bool, int]:
        """Return `(is_durable, max_attempts)` for `node_name` on `graph`."""
        if _is_tool_or_subgraph_node(graph, node_name):
            return False, max(1, int(limits.retries))
        bare_node = node_name.rsplit(":", 1)[-1] if ":" in node_name else node_name
        policy = self.policy or {}
        candidates = _node_candidates(graph, bare_node)
        marked_durable: bool | None = None
        marked_retries: int | None = None
        for cand in candidates:
            if marked_durable is None and "_durable_node" in getattr(
                cand, "__dict__", {}
            ):
                marked_durable = bool(getattr(cand, "_durable_node"))
            if marked_retries is None and "_durable_retries" in getattr(
                cand, "__dict__", {}
            ):
                marked_retries = int(getattr(cand, "_durable_retries"))

        if node_name in policy:
            is_durable = policy[node_name]
        elif bare_node in policy:
            is_durable = policy[bare_node]
        elif marked_durable is not None:
            is_durable = marked_durable
        else:
            is_durable = self.default_durable

        retries_map = self.retries or {}
        if node_name in retries_map:
            attempts = retries_map[node_name]
        elif bare_node in retries_map:
            attempts = retries_map[bare_node]
        elif marked_retries is not None:
            attempts = marked_retries
        elif self.default_retries is not None:
            attempts = self.default_retries
        elif limits.node_retries is not None:
            attempts = limits.node_retries
        else:
            attempts = limits.retries
        return is_durable, max(1, int(attempts))


def _effective_node_config(
    graph: CompiledGraph, node_config: NodeConfig | None = None
) -> NodeConfig:
    """Return the explicit `node_config`, the graph's attached config, or a default.

    `NodeConfig(default_durable=False)`.
    """
    if node_config is not None:
        return node_config
    cfg = getattr(graph, "_durable_node_config", None)
    if isinstance(cfg, NodeConfig):
        return cfg
    return NodeConfig(default_durable=False)


def _selector_key(selector: Selector | Any) -> str:
    if isinstance(selector, str):
        return selector
    tool_attr = getattr(selector, "tool", None)
    if tool_attr is not None:
        node = getattr(selector, "node", None)
        return f"{node}::{tool_attr}" if node else str(tool_attr)
    if isinstance(selector, tuple) and len(selector) == 2:
        return f"{selector[0]}::{selector[1]}"
    return str(selector)


def _parse_policy_value(value: PolicyValue | str) -> tuple[bool, int | None]:
    if isinstance(value, bool):
        return value, None
    if isinstance(value, int):
        return value > 0, value
    s = str(getattr(value, "value", value)).lower()
    if s in (ExecutionMode.DURABLE, "true"):
        return True, None
    if s in (ExecutionMode.LOCAL, "false"):
        return False, None
    raise ValueError(f"unsupported tool policy value {value!r}")
