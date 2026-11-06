# Copyright (c) Microsoft. All rights reserved.

"""LangGraph Pregel state inspection, composite checkpoint IDs, and single-tool.

execution.

Driving a compiled LangGraph `Pregel` graph one durable superstep at a time from
an external orchestrator requires bridging four non-obvious LangGraph internal
behaviors:

1. **Composite Checkpoint IDs (`_checkpoint_id` -> `<uuid>@<12-hex-hash>`):**
   LangGraph only mints a new checkpoint UUID when a superstep finishes. Intra-step
   mutations (sequential `interrupt()` calls inside one node, subgraph progress, or
   single-branch writes in a parallel superstep) keep the same `checkpoint_id`. We
   append a 12-hex-char SHA-256 digest of those intra-step states so idempotency
   guards can tell whether a step actually advanced.
2. **Mid-Start Crash Recovery (`_is_mid_start`):**
   `graph.invoke(input)` writes a `source="input"` checkpoint *before* Node 1 runs.
   If a worker crashes inside Node 1, retrying with `input` would duplicate input
   messages, while treating the new checkpoint ID as "already completed" would skip
   Node 1. `_is_mid_start` detects this exact window so retry calls `invoke(None)`.
3. **Parallel Branch Writes (`_put_task_writes` vs `_update_state`):**
   `graph.update_state(as_node=...)` creates a brand-new superstep checkpoint that
   cancels any sibling parallel branches scheduled in `before.next`. When a
   `ToolNode` runs in parallel with another node, we record its output via
   `saver.put_writes()` so the sibling branch still runs in the next step.
4. **Isolated Durable Tool Execution (`_exec_single_tool`):**
   Executing one tool call outside `ToolNode.invoke()` requires synthesizing a
   `PregelScratchpad` on `var_child_runnable_config` so `interrupt()`, `Command`,
   and `InjectedState` / `ToolRuntime` work identically to native LangGraph.
"""

from __future__ import annotations

import asyncio
import builtins
import copy
import hashlib
import inspect
import itertools
import json
import weakref
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, cast

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableSequence
from langchain_core.runnables.config import var_child_runnable_config
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_READ,
    CONFIG_KEY_RUNTIME,
    CONFIG_KEY_SCRATCHPAD,
    CONFIG_KEY_SEND,
    CONFIG_KEY_TASK_ID,
    INPUT,
    NO_WRITES,
    NULL_TASK_ID,
)
from langgraph._internal._runnable import patch_config
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.errors import GraphInterrupt
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import (
    ToolRuntime,
    _default_handle_tool_errors,
    _handle_tool_error,
)
from langgraph.pregel._algo import PregelTaskWrites, apply_writes, prepare_next_tasks
from langgraph.pregel._checkpoint import channels_from_checkpoint
from langgraph.runtime import DEFAULT_RUNTIME
from langgraph.types import Command, Overwrite, PregelExecutableTask, StateSnapshot

from langchain_azure_compute.graph._async_bridge import (
    _ensure_async_saver,
    _is_async_saver,
    _run_async,
)
from langchain_azure_compute.graph._config import (
    STATIC_AFTER_PREFIX,
    STATIC_BEFORE_PREFIX,
    CompiledGraph,
    InterruptEntry,
    Limits,
    NodeConfig,
    PackedBlob,
    RunError,
    TaskResultStatus,
    ToolConfig,
    pack,
    unpack,
)
from langchain_azure_compute.graph._inline_saver import (
    _export_checkpoint_bundle,
    _hydrate_checkpoint_bundle,
)

_INTERNAL_CONF_KEYS = {"thread_id", "checkpoint_id", "checkpoint_ns"}


def _null_resume(_consume: bool = False) -> Any:
    return None


class _StepRecursionLimit(int):
    """Step-adjusted `int` for `config['recursion_limit']` across superstep activities.

    LangGraph's `PregelLoop.__enter__` computes `self.stop = self.step +
    config['recursion_limit'] + 1`
    on every `graph.invoke()` call. When stepping one superstep at a time
    (`interrupt_after="*"`),
    passing the unadjusted `recursion_limit` would push `self.stop` forward on every
    step, preventing
    `RemainingSteps` from counting down and `IsLastStep` from ever becoming `True`.
    This `int` subclass supplies the remaining step delta for `self.stop` arithmetic
    while comparing
    and formatting as the user's original `total` recursion limit in `Pregel._defaults`
    and
    `GraphRecursionError` messages.
    """

    total: int

    def __new__(cls, adj: int, total: int) -> _StepRecursionLimit:
        obj = super().__new__(cls, adj)
        obj.total = total
        return obj

    def __lt__(self, other: object) -> bool:
        return (
            int(self.total) < other
            if isinstance(other, (int, float))
            else NotImplemented
        )

    def __str__(self) -> str:
        return str(self.total)  # pragma: no cover

    def __format__(self, spec: str) -> str:
        return format(self.total, spec)


def _normalize_run_config(
    config: RunnableConfig | Mapping[str, Any] | None,
) -> RunnableConfig | None:
    """Extract serializable user `RunnableConfig` fields (`configurable`, `tags`.

    `metadata`, `recursion_limit`).
    """
    if not config or not isinstance(config, Mapping):
        return None
    out: dict[str, Any] = {}
    conf = config.get("configurable")
    if isinstance(conf, Mapping):  # pragma: no cover
        user_conf = {
            k: v
            for k, v in conf.items()
            if k not in _INTERNAL_CONF_KEYS and not str(k).startswith("__pregel_")
        }
        if user_conf:  # pragma: no cover
            out["configurable"] = user_conf
    if config.get("tags"):  # pragma: no cover
        out["tags"] = list(config["tags"])
    if isinstance(config.get("metadata"), Mapping) and config["metadata"]:
        out["metadata"] = dict(config["metadata"])
    if config.get("recursion_limit") is not None:  # pragma: no cover
        out["recursion_limit"] = int(config["recursion_limit"])  # pragma: no cover
    return cast(RunnableConfig, out) if out else None


def _state_schema_for_graph(
    graph: CompiledGraph, node_name: str = ""
) -> type[Any] | None:
    """Resolve the `state_schema` for `graph` (or for the nested subgraph when.

    `node_name` is scoped).
    """
    target: Any = graph
    if ":" in node_name and callable(getattr(graph, "get_subgraphs", None)):
        prefix = node_name.rsplit(":", 1)[0]
        try:
            for pfx, sub in graph.get_subgraphs(recurse=True):  # pragma: no cover
                if pfx == prefix:  # pragma: no cover
                    target = sub
                    break
        except Exception:  # pragma: no cover
            pass  # pragma: no cover
    builder = getattr(target, "builder", None)
    return getattr(builder, "state_schema", None) or getattr(builder, "schema", None)


def _coerce_with_schema(value: Any, schema: type[Any] | None) -> Any:
    """Coerce a `dict` state or context into `schema` when `schema` is a `BaseModel` or.

    `@dataclass`.
    """
    if (
        isinstance(value, dict)
        and isinstance(schema, type)
        and not issubclass(schema, dict)
        and not isinstance(value, schema)
    ):
        try:
            return schema(**value)
        except Exception:  # pragma: no cover
            return value  # pragma: no cover
    return value


def _pack_context(context: Any) -> PackedBlob:
    """Serialize `context` for DTS transport, converting `@dataclass` or `BaseModel`.

    instances to dicts.
    """
    if is_dataclass(context) and not isinstance(context, type):  # pragma: no cover
        return pack(asdict(context))
    if hasattr(context, "model_dump") and callable(
        context.model_dump
    ):  # pragma: no cover
        return pack(context.model_dump())  # pragma: no cover
    return pack(context)  # pragma: no cover


def _unpack_context(graph: CompiledGraph, blob: PackedBlob | Mapping[str, Any]) -> Any:
    """Deserialize `blob` and hydrate into `graph.context_schema` when configured."""
    val = unpack(blob)
    schema = getattr(graph, "context_schema", None) or getattr(
        getattr(graph, "builder", None), "context_schema", None
    )
    return _coerce_with_schema(val, schema)


def _tool_node_handles_errors(tn: ToolNode | Any) -> bool:
    """Return True if `tn` is configured to catch runtime tool exceptions into error.

    `ToolMessage`s.
    """
    flag = getattr(tn, "_handle_tool_errors", False)
    return bool(flag) and flag is not _default_handle_tool_errors


def _format_failed_tool_message(
    tn: ToolNode | Any, call: Mapping[str, Any], err_info: Mapping[str, Any]
) -> ToolMessage:
    """Convert an exhausted durable tool failure into a `ToolMessage(status='error')`.

    via `ToolNode`'s error policy.
    """
    raw_type = str(err_info.get("error_type") or "Exception").rsplit(".", 1)[-1]
    msg = str(err_info.get("message") or "")
    exc_cls = getattr(builtins, raw_type, None)
    if not (
        isinstance(exc_cls, type) and issubclass(exc_cls, Exception)
    ):  # pragma: no cover
        exc_cls = RuntimeError  # pragma: no cover
    exc_obj = exc_cls(msg)
    flag = getattr(tn, "_handle_tool_errors", True)
    if flag is _default_handle_tool_errors:  # pragma: no cover
        flag = True  # pragma: no cover
    if isinstance(flag, type) and issubclass(flag, Exception):  # pragma: no cover
        if not isinstance(exc_obj, flag):  # pragma: no cover
            raise exc_obj  # pragma: no cover
    elif isinstance(flag, tuple):  # pragma: no cover
        if not isinstance(exc_obj, flag):  # pragma: no cover
            raise exc_obj  # pragma: no cover
    content = _handle_tool_error(exc_obj, flag=flag)
    return ToolMessage(
        content=content,
        name=call.get("name"),
        tool_call_id=call.get("id"),
        status="error",
    )


def _config(
    thread: str,
    checkpoint_id: str | None = None,
    checkpoint_ns: str = "",
    run_config: RunnableConfig | Mapping[str, Any] | None = None,
) -> RunnableConfig:
    """Build a LangGraph `RunnableConfig` dict targeting `thread` (and optional.

    checkpoint/namespace/user config).
    """
    cfg: dict[str, Any] = {}
    if run_config and isinstance(run_config.get("configurable"), Mapping):
        cfg.update(run_config["configurable"])
    cfg["thread_id"] = thread
    cfg["checkpoint_ns"] = checkpoint_ns
    if checkpoint_id is not None:
        cfg["checkpoint_id"] = checkpoint_id
    out: dict[str, Any] = {"configurable": cfg}
    if run_config:
        if run_config.get("tags"):
            out["tags"] = list(run_config["tags"])
        if run_config.get("metadata"):
            out["metadata"] = dict(run_config["metadata"])
        if run_config.get("recursion_limit") is not None:
            out["recursion_limit"] = int(run_config["recursion_limit"])
    return cast(RunnableConfig, out)


def _checkpoint_id(
    snapshot_or_config: StateSnapshot | RunnableConfig | Mapping[str, Any] | None,
) -> str | None:
    """Return a composite `<uuid>@<12-hex-hash>` identifier for the current graph state.

    When a single node calls `interrupt()` multiple times sequentially (e.g., asking for
    confirmation #1, receiving resume #1, and then calling `interrupt()` #2 before the
    node returns), LangGraph re-executes the node from the top and keeps the *same*
    base `checkpoint_id` UUID until the node completes. The same UUID reuse also happens
    when a nested subgraph advances internally or when one branch of a parallel
    superstep
    records pending writes via `put_writes()`.

    If `run_step` only compared the raw `checkpoint_id` UUID against `expect`, a crash
    or retry after interrupt #1 would see the same UUID and mistakenly treat interrupt
    #2's
    resume as already replayed (or vice versa). Appending a 12-character SHA-256 digest
    of:
      - pending task interrupts (`i`),
      - nested subgraph checkpoint states (`s`),
      - completed parallel task writes (`d`), and
      - scheduled `next` nodes (`n`)
    uniquely identifies every observable sub-step state while stripping cleanly via
    `_raw_checkpoint_id()` whenever LangGraph needs the underlying UUID.
    """
    if snapshot_or_config is None:  # pragma: no cover
        return None  # pragma: no cover
    cfg: Any = getattr(snapshot_or_config, "config", snapshot_or_config)
    base_id = (cfg or {}).get("configurable", {}).get("checkpoint_id")
    if base_id is None or not hasattr(snapshot_or_config, "tasks"):
        return base_id
    tasks = getattr(snapshot_or_config, "tasks", None) or ()
    intrs = [
        (t.id, i.id, pack(i.value)) for t in tasks for i in getattr(t, "interrupts", ())
    ]
    sub_states = [
        (t.id, _checkpoint_id(t.state), tuple(getattr(t.state, "next", ()) or ()))
        for t in tasks
        if getattr(t, "state", None) is not None
    ]
    nxt = tuple(getattr(snapshot_or_config, "next", ()) or ())
    done_tasks = [t.id for t in tasks if getattr(t, "result", None) is not None]
    meta = getattr(snapshot_or_config, "metadata", None) or {}
    mig_to = meta.get("migrated_to")
    if not intrs and not sub_states and not done_tasks and mig_to is None:
        return base_id
    digest_dict: dict[str, Any] = {
        "i": intrs,
        "s": sub_states,
        "d": done_tasks,
        "n": nxt,
    }
    if mig_to is not None:
        digest_dict["m"] = mig_to
    digest_input = json.dumps(
        digest_dict,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    h = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
    return f"{base_id}@{h}"


def _raw_checkpoint_id(
    snapshot_or_config: StateSnapshot | RunnableConfig | Mapping[str, Any] | str | None,
) -> str | None:
    """Extract the bare LangGraph UUID from a composite `<uuid>@<hash>` checkpoint ID.

    or snapshot.
    """
    if snapshot_or_config is None:
        return None
    if isinstance(snapshot_or_config, str):
        return snapshot_or_config.split("@", 1)[0]
    cfg: Any = getattr(snapshot_or_config, "config", snapshot_or_config)
    val = (cfg or {}).get("configurable", {}).get("checkpoint_id")
    return val.split("@", 1)[0] if isinstance(val, str) else val


def _checkpoint_match(snapshot: StateSnapshot | None, expect: str | None) -> bool:
    """Return True if `snapshot` matches the expected composite or raw checkpoint ID."""
    cur = _checkpoint_id(snapshot)
    if cur == expect:
        return True
    if expect is not None and "@" not in expect and _raw_checkpoint_id(cur) == expect:
        meta = getattr(snapshot, "metadata", None) or {}
        if meta.get("migrated_to") is not None:
            return False
        return True
    return False


def _parent_checkpoint_id(snapshot: StateSnapshot | None) -> str | None:
    """Return the raw UUID of `snapshot.parent_config`, or None if at the root."""
    return _raw_checkpoint_id(getattr(snapshot, "parent_config", None))


def _get_state(
    graph: CompiledGraph, config: RunnableConfig, *, subgraphs: bool = False
) -> StateSnapshot:
    """Read `StateSnapshot` from `graph`, bridging to the background event loop for.

    async savers.
    """
    saver = getattr(graph, "checkpointer", None)
    if _is_async_saver(saver):
        return _run_async(graph, graph.aget_state(config, subgraphs=subgraphs))
    try:
        return graph.get_state(config, subgraphs=subgraphs)
    except (NotImplementedError, asyncio.InvalidStateError):  # pragma: no cover
        _ensure_async_saver(saver)  # pragma: no cover
        return _run_async(
            graph, graph.aget_state(config, subgraphs=subgraphs)
        )  # pragma: no cover


def _update_state(
    graph: CompiledGraph,
    config: RunnableConfig,
    values: dict[str, Any] | Any | None,
    as_node: str | None = None,
) -> RunnableConfig:
    """Commit `values` as `as_node` via `graph.update_state` (or `aupdate_state` for.

    async savers).
    """
    saver = getattr(graph, "checkpointer", None)
    if _is_async_saver(saver):  # pragma: no cover
        return _run_async(
            graph, graph.aupdate_state(config, values, as_node=as_node)
        )  # pragma: no cover
    try:
        return graph.update_state(config, values, as_node=as_node)
    except (NotImplementedError, asyncio.InvalidStateError):  # pragma: no cover
        _ensure_async_saver(saver)  # pragma: no cover
        return _run_async(
            graph, graph.aupdate_state(config, values, as_node=as_node)
        )  # pragma: no cover


def _put_task_writes(
    graph: CompiledGraph,
    snapshot: StateSnapshot,
    node_name: str,
    task_id: str,
    values: Any,
) -> None:
    """Record pending writes for a single parallel task without cancelling sibling.

    tasks.

    When a superstep fans out to multiple nodes in parallel (for example, `before.next`
    is
    `("tools", "audit_branch")`), our `interrupt_before=["tools"]` pauses the graph
    right
    before `"tools"` runs so we can execute its tools as individual DTS activities.
    Once those tool activities finish, we must commit the `ToolNode`'s output back to
    the
    checkpoint. Calling `graph.update_state(config, combined, as_node="tools")` would
    create a brand-new checkpoint whose `next` transitions only from `"tools"`, silently
    dropping `"audit_branch"` from the current superstep.
    Instead, running the node's channel writers (`flat_writers`) and persisting them via
    `saver.put_writes(..., task_id)` marks *only* the `"tools"` task as completed inside
    the existing checkpoint, so the next `graph.invoke(None)` runs `"audit_branch"` and
    then joins both branches cleanly.
    """
    writers = graph.nodes[node_name].flat_writers
    writes: deque[tuple[str, Any]] = deque()
    schema = _state_schema_for_graph(graph, node_name)

    def _read(select: str | Sequence[str], fresh: bool = False) -> Any:
        base = (
            copy.deepcopy(snapshot.values)
            if isinstance(snapshot.values, dict)
            else {"__root__": copy.deepcopy(snapshot.values)}
        )
        if fresh:  # pragma: no cover
            channels = getattr(graph, "channels", None) or {}
            written_keys: list[str] = []
            for ch, _ in writes:
                if ch not in written_keys:  # pragma: no cover
                    written_keys.append(ch)
            for ch in written_keys:
                ch_vals = [val for k, val in writes if k == ch]
                ch_obj = channels.get(ch)
                if ch_obj is not None and hasattr(ch_obj, "copy"):  # pragma: no cover
                    try:
                        c = (
                            ch_obj.from_checkpoint(base[ch])
                            if ch in base and hasattr(ch_obj, "from_checkpoint")
                            else ch_obj.copy()
                        )
                        c.update(ch_vals)
                        base[ch] = c.get()
                        continue
                    except Exception:  # pragma: no cover
                        pass  # pragma: no cover
                base[ch] = ch_vals[-1]  # pragma: no cover
        if isinstance(select, str):  # pragma: no cover
            return (  # pragma: no cover
                _coerce_with_schema(base, schema)
                if select == "__root__"
                else base.get(select)
            )
        return {k: base.get(k) for k in select if k in base}

    scratchpad = PregelScratchpad(
        step=0,
        stop=1,
        call_counter=itertools.count(0).__next__,
        interrupt_counter=itertools.count(0).__next__,
        get_null_resume=_null_resume,
        resume=[],
        subgraph_counter=itertools.count(0).__next__,
    )
    run: Any = RunnableSequence(*writers) if len(writers) > 1 else writers[0]
    run.invoke(
        values,
        patch_config(
            snapshot.config,
            configurable={
                CONFIG_KEY_SEND: writes.extend,
                CONFIG_KEY_TASK_ID: task_id,
                CONFIG_KEY_READ: _read,
                "__pregel_scratchpad": scratchpad,
            },
        ),
    )
    if not writes:  # pragma: no cover
        writes.append((NO_WRITES, None))  # pragma: no cover
    saver: Any = getattr(graph, "checkpointer", None)
    if saver is None:  # pragma: no cover
        return  # pragma: no cover
    if _is_async_saver(saver):  # pragma: no cover
        _run_async(
            graph, saver.aput_writes(snapshot.config, list(writes), task_id)
        )  # pragma: no cover
    else:
        try:
            saver.put_writes(snapshot.config, list(writes), task_id)
        except (NotImplementedError, asyncio.InvalidStateError):  # pragma: no cover
            _ensure_async_saver(saver)  # pragma: no cover
            _run_async(
                graph, saver.aput_writes(snapshot.config, list(writes), task_id)
            )  # pragma: no cover


def _resolve_turn_input_step(
    graph: CompiledGraph,
    config: RunnableConfig,
    turn_input_step: int | None = None,
) -> int:
    """Resolve the checkpointer step index where the current turn's input was written."""  # noqa: E501
    if turn_input_step is not None:  # pragma: no cover
        return int(turn_input_step)  # pragma: no cover
    try:
        cur_cfg: Any = config
        for _ in range(128):  # pragma: no cover
            if not cur_cfg:  # pragma: no cover
                break  # pragma: no cover
            snap = _get_state(graph, cur_cfg)
            meta = getattr(snap, "metadata", None) or {}
            if meta.get("source") == "input":
                return int(meta.get("step", -1))
            cur_cfg = getattr(snap, "parent_config", None)
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    return -1  # pragma: no cover


def _graph_has_async_only_node(graph: CompiledGraph) -> bool:
    """Return True if any node in `graph` only provides an async callable."""
    builder = getattr(graph, "builder", None)
    nodes = getattr(builder, "nodes", None) or {}
    for spec in nodes.values():
        runnable = getattr(spec, "runnable", spec)
        if (
            getattr(runnable, "func", "present") is None
            and getattr(runnable, "afunc", None) is not None
        ):
            return True
    for node_obj in (getattr(graph, "nodes", None) or {}).values():
        bound = getattr(node_obj, "bound", node_obj)
        if (
            getattr(bound, "func", "present") is None
            and getattr(bound, "afunc", None) is not None
        ):  # pragma: no cover
            return True  # pragma: no cover
    return False


def _invoke(
    graph: CompiledGraph,
    inp: Any,
    config: RunnableConfig,
    *,
    interrupt_after: Sequence[str] | Literal["*"] = "*",
    interrupt_before: Sequence[str] | Literal["*"] | None = None,
    max_concurrency: int | None = None,
    context: Any = None,
    before_step: int | None = None,
    turn_input_step: int | None = None,
) -> Any:
    """Invoke `graph` for a single superstep (`interrupt_after="*"`), falling back to.

    `ainvoke` when needed.
    """
    cfg_dict: dict[str, Any] = dict(config)
    if max_concurrency is not None:
        cfg_dict["max_concurrency"] = int(max_concurrency)
    has_managed = bool(getattr(getattr(graph, "builder", None), "managed", None))
    if "recursion_limit" in cfg_dict or has_managed:
        total_limit = int(cfg_dict.get("recursion_limit", 25))
        if inp is not None and not isinstance(inp, Command):
            cfg_dict["recursion_limit"] = total_limit
        else:
            cur_step = before_step if before_step is not None else -2
            turn_in_step = _resolve_turn_input_step(graph, config, turn_input_step)
            adj_limit = total_limit - (cur_step + 1 - turn_in_step)
            cfg_dict["recursion_limit"] = _StepRecursionLimit(adj_limit, total_limit)
    eff_cfg = cast(RunnableConfig, cfg_dict)
    kwargs: dict[str, Any] = {"interrupt_after": interrupt_after}
    kwargs["interrupt_before"] = interrupt_before or []
    if context is not None:
        kwargs["context"] = context
    saver = getattr(graph, "checkpointer", None)
    if _is_async_saver(saver) or _graph_has_async_only_node(graph):
        _ensure_async_saver(saver)
        return _run_async(graph, graph.ainvoke(inp, eff_cfg, **kwargs))
    try:
        return graph.invoke(inp, eff_cfg, **kwargs)
    except (NotImplementedError, asyncio.InvalidStateError):
        _ensure_async_saver(saver)  # pragma: no cover
        retry_inp = inp  # pragma: no cover
        if inp is not None and not isinstance(inp, Command):  # pragma: no cover
            try:  # pragma: no cover
                cur_snap = _get_state(graph, config)  # pragma: no cover
                if (getattr(cur_snap, "metadata", None) or {}).get("source") in (
                    "input",
                    "loop",
                ):  # pragma: no cover
                    retry_inp = None  # pragma: no cover
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        return _run_async(
            graph, graph.ainvoke(retry_inp, eff_cfg, **kwargs)
        )  # pragma: no cover
    except TypeError as exc:
        if "No synchronous function provided" not in str(exc):  # pragma: no cover
            raise
        _ensure_async_saver(saver)  # pragma: no cover
        retry_inp = inp  # pragma: no cover
        if inp is not None and not isinstance(inp, Command):  # pragma: no cover
            try:  # pragma: no cover
                cur_snap = _get_state(graph, config)  # pragma: no cover
                if (getattr(cur_snap, "metadata", None) or {}).get("source") in (
                    "input",
                    "loop",
                ):  # pragma: no cover
                    retry_inp = None  # pragma: no cover
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        return _run_async(
            graph, graph.ainvoke(retry_inp, eff_cfg, **kwargs)
        )  # pragma: no cover


def _is_mid_start(
    graph: CompiledGraph,
    snapshot: StateSnapshot,
    expect: str | None,
    interrupt_before: Sequence[str] | str = (),
) -> bool:
    """Detect whether a `"start"` activity crashed after writing the input checkpoint.

    but before Node 1 finished.

    When `graph.invoke(input, interrupt_after="*")` starts a new turn from checkpoint
    `expect`,
    LangGraph writes checkpoints in this 3-phase timeline:
      1. `step=-1, source="input"`  - persists the user `input` channels (parent ==
         `expect`)
      2. `step=0,  source="loop"`   - advances `__start__` so `next = (Node_1,)` (parent
         == `step=-1`)
      3. `Node_1` executes and writes `step=1, source="loop"` after `Node_1` finishes.

    If the worker crashes during phase 1, phase 2, or mid-execution inside `Node_1`, the
    checkpointer's latest checkpoint is no longer `expect` - even though `Node_1` has
    not
    completed. On DTS activity retry:
      - If we assumed `before != expect` meant the step already finished, we would skip
        `Node_1`.
      - If we called `graph.invoke(input)` again, we would duplicate the user's input
        messages
        (e.g., appending a second `HumanMessage` via `add_messages`).
    By checking whether the current checkpoint (or its immediate parent before an unhit
    `interrupt_before`) is the `source="input"` child of `expect`, `run_step` knows to
    resume
    via `graph.invoke(None)` so `Node_1` runs once without duplicating `input`.
    """
    raw_expect = _raw_checkpoint_id(expect)
    meta = getattr(snapshot, "metadata", None) or {}
    if (
        meta.get("source") == "input" and _parent_checkpoint_id(snapshot) == raw_expect
    ):  # pragma: no cover
        return True  # pragma: no cover
    nxt = tuple(getattr(snapshot, "next", ()) or ())
    hit_before = (
        bool(nxt) if interrupt_before == "*" else bool(set(nxt) & set(interrupt_before))
    )
    if meta.get("source") == "loop" and nxt and not hit_before:
        parent_cfg = getattr(snapshot, "parent_config", None)
        if parent_cfg:  # pragma: no cover
            parent = _get_state(graph, parent_cfg)
            pmeta = getattr(parent, "metadata", None) or {}
            if (
                pmeta.get("source") == "input"
                and _parent_checkpoint_id(parent) == raw_expect
            ):  # pragma: no cover
                return True
    return False


def read_output(
    graph: CompiledGraph, thread: str, checkpoint_id: str | None = None
) -> Any:
    """Read the graph's state from the checkpointer and apply `output_channels`.

    projection if configured.
    """
    graph = _unwrap_registered_graph(graph)
    raw_cid = _raw_checkpoint_id(checkpoint_id) if checkpoint_id is not None else None
    if raw_cid is not None:
        snap = _get_state(graph, _config(thread, raw_cid))
        if snap is not None and getattr(snap, "values", None):  # pragma: no cover
            return _project_output(graph, snap.values)
    snap = _get_state(graph, _config(thread))
    return _project_output(graph, snap.values)


def _project_output(graph: CompiledGraph, values: Any) -> Any:
    """Filter `values` down to `graph.output_channels` when the graph defines an output.

    schema.
    """
    out_channels = getattr(graph, "output_channels", None)
    if isinstance(out_channels, (list, tuple)) and isinstance(
        values, dict
    ):  # pragma: no cover
        return {k: values[k] for k in out_channels if k in values}
    return values  # pragma: no cover


def _tool_nodes(graph: CompiledGraph) -> dict[str, ToolNode]:
    """Return a map of `{node_name: ToolNode}` for top-level and `prefix:node` subgraph.

    `ToolNode`s.
    """
    builder = getattr(graph, "builder", None)
    nodes = getattr(builder, "nodes", None) or {}
    out: dict[str, ToolNode] = {}
    for name, spec in nodes.items():
        runnable = getattr(spec, "runnable", spec)
        if hasattr(runnable, "tools_by_name") and hasattr(runnable, "_parse_input"):
            out[name] = runnable
    get_subgraphs = getattr(graph, "get_subgraphs", None)
    if callable(get_subgraphs):  # pragma: no cover
        try:
            for prefix, sub in get_subgraphs(recurse=True):
                sbuilder = getattr(sub, "builder", None)
                snodes = getattr(sbuilder, "nodes", None) or {}
                for sname, sspec in snodes.items():
                    srun = getattr(sspec, "runnable", sspec)
                    if hasattr(srun, "tools_by_name") and hasattr(srun, "_parse_input"):
                        out[f"{prefix}:{sname}"] = srun
        except Exception:  # pragma: no cover
            pass  # pragma: no cover
    return out


def _prepare_subgraph_tool_nodes(
    graph: CompiledGraph, limits: Limits, tool_config: ToolConfig
) -> None:
    """Configure breakpoints before durable tool nodes in nested subgraphs.

    LangGraph pauses before the ToolNode and yields control to the orchestrator.
    """
    get_subgraphs = getattr(graph, "get_subgraphs", None)
    if not callable(get_subgraphs):  # pragma: no cover
        return  # pragma: no cover
    for prefix, sub in get_subgraphs(recurse=True):
        if not hasattr(sub, "_durable_orig_interrupt_before"):
            orig_sub_ub = getattr(sub, "interrupt_before_nodes", None)
            setattr(
                sub,
                "_durable_orig_interrupt_before",
                "*" if orig_sub_ub == "*" else tuple(orig_sub_ub or ()),
            )
        base_ub = getattr(sub, "_durable_orig_interrupt_before")
        builder = getattr(sub, "builder", None)
        nodes = getattr(builder, "nodes", None) or {}
        dur_names: list[str] = []
        for node_name, spec in nodes.items():
            runnable = getattr(spec, "runnable", spec)
            if hasattr(runnable, "tools_by_name") and hasattr(runnable, "_parse_input"):
                tools_by_name = getattr(runnable, "tools_by_name", {}) or {}
                if any(
                    tool_config.resolve(
                        runnable, f"{prefix}:{node_name}", tname, limits
                    )[0]
                    or tool_config.resolve(runnable, node_name, tname, limits)[0]
                    for tname in tools_by_name
                ):  # pragma: no cover
                    dur_names.append(node_name)
        if base_ub == "*":  # pragma: no cover
            sub.interrupt_before_nodes = "*"  # pragma: no cover
        else:
            merged = list(base_ub)
            for n in dur_names:
                if n not in merged:  # pragma: no cover
                    merged.append(n)
            sub.interrupt_before_nodes = merged


def _durable_tool_node_names(
    graph: CompiledGraph,
    limits: Limits,
    tool_config: ToolConfig,
    snapshot: StateSnapshot | None = None,
) -> list[str]:
    """Return unfinished top-level tool nodes with durable tools.

    Only nodes pending in the current parallel superstep are included.
    """
    names: list[str] = []
    builder = getattr(graph, "builder", None)
    top_nodes = getattr(builder, "nodes", None) or {}
    tasks = getattr(snapshot, "tasks", None) or ()
    uncompleted_nodes = {t.name for t in tasks if getattr(t, "result", None) is None}
    completed_tasks = {
        t.name
        for t in tasks
        if getattr(t, "result", None) is not None and t.name not in uncompleted_nodes
    }
    snap_next = (
        set(getattr(snapshot, "next", None) or ()) if snapshot is not None else set()
    )
    for node_name, spec in top_nodes.items():
        if node_name in completed_tasks:
            continue
        tn = getattr(spec, "runnable", spec)
        if hasattr(tn, "tools_by_name") and hasattr(tn, "_parse_input"):
            tools_by_name = getattr(tn, "tools_by_name", {}) or {}
            if not any(
                tool_config.resolve(tn, node_name, tname, limits)[0]
                for tname in tools_by_name
            ):
                continue
            if node_name in snap_next and snapshot is not None:
                try:
                    tcalls, _ = tn._parse_input(snapshot.values)
                except Exception:  # pragma: no cover
                    tcalls = []  # pragma: no cover
                if tcalls and not any(
                    tool_config.resolve(tn, node_name, c["name"], limits)[0]
                    for c in tcalls
                ):  # pragma: no cover
                    # All currently pending tool calls on this ToolNode are marked local
                    # (durable=False),
                    # so allow the next step to execute this ToolNode inline instead of
                    # pausing before it again.
                    continue  # pragma: no cover
            names.append(node_name)
    return names


def _durable_node_names(
    graph: CompiledGraph,
    limits: Limits,
    node_config: NodeConfig,
    snapshot: StateSnapshot | None = None,
) -> list[str]:
    """Return unfinished top-level nodes configured as durable activities.

    Includes @durable_node and durable_nodes policies for the current superstep.
    """
    names: list[str] = []
    builder = getattr(graph, "builder", None)
    top_nodes = getattr(builder, "nodes", None) or {}
    tasks = getattr(snapshot, "tasks", None) or ()
    uncompleted_nodes = {t.name for t in tasks if getattr(t, "result", None) is None}
    completed_tasks = {
        t.name
        for t in tasks
        if getattr(t, "result", None) is not None and t.name not in uncompleted_nodes
    }
    for node_name in top_nodes:
        if node_name in completed_tasks:
            continue
        if node_config.resolve(graph, node_name, limits)[0]:
            names.append(node_name)
    return names


def _user_interrupt_before(graph: CompiledGraph) -> tuple[str, ...] | Literal["*"]:
    if hasattr(graph, "_durable_user_interrupt_before"):
        return getattr(graph, "_durable_user_interrupt_before")
    ub = getattr(graph, "interrupt_before_nodes", None)
    return "*" if ub == "*" else tuple(ub or ())


def _user_interrupt_after(graph: CompiledGraph) -> tuple[str, ...] | Literal["*"]:
    if hasattr(graph, "_durable_user_interrupt_after"):
        return getattr(graph, "_durable_user_interrupt_after")
    ua = getattr(graph, "interrupt_after_nodes", None)
    return "*" if ua == "*" else tuple(ua or ())


def _merged_interrupt_before(
    graph: CompiledGraph,
    durable_tnodes: Sequence[str],
    *,
    exclude_static: Sequence[str] = (),
) -> list[str] | Literal["*"]:
    """Combine user-configured `interrupt_before` nodes with internal durable.

    `ToolNode` pause points.
    """
    ub = _user_interrupt_before(graph)
    if ub == "*":  # pragma: no cover
        if exclude_static:  # pragma: no cover
            builder = getattr(graph, "builder", None)  # pragma: no cover
            all_nodes = list(
                (getattr(builder, "nodes", None) or {}).keys()
            )  # pragma: no cover
            ub_list = [
                n for n in all_nodes if n not in exclude_static
            ]  # pragma: no cover
        else:
            return "*"  # pragma: no cover
    else:
        ub_list = [n for n in ub if n not in exclude_static]
    merged = list(durable_tnodes)
    for n in ub_list:
        if n not in merged:  # pragma: no cover
            merged.append(n)
    return merged


def _find_subgraph_tool_target(
    snapshot: StateSnapshot, tnodes: Mapping[str, ToolNode], prefix: str = ""
) -> tuple[str, str, StateSnapshot, str] | None:
    """Recursively inspect `snapshot.tasks` to find a paused `ToolNode` inside a nested.

    subgraph.
    """
    for t in getattr(snapshot, "tasks", ()) or ():
        sub_snap = getattr(t, "state", None)
        if sub_snap is None:
            continue
        cur_prefix = f"{prefix}:{t.name}" if prefix else t.name
        for n in getattr(sub_snap, "next", ()) or ():
            scoped_name = f"{cur_prefix}:{n}"
            if scoped_name in tnodes:
                ns = (
                    (sub_snap.config or {})
                    .get("configurable", {})
                    .get("checkpoint_ns", "")
                )
                sub_ckpt = _raw_checkpoint_id(sub_snap) or ""
                return scoped_name, n, sub_snap, f"{ns}|{sub_ckpt}"
        nested = _find_subgraph_tool_target(sub_snap, tnodes, cur_prefix)
        if nested is not None:  # pragma: no cover
            return nested  # pragma: no cover
    return None


def _static_breakpoint_hit(
    graph: CompiledGraph, snapshot: StateSnapshot
) -> InterruptEntry | None:
    """Return a synthetic interrupt descriptor for a user-configured static breakpoint.

    Checks both interrupt_before and interrupt_after pauses in the snapshot.
    """
    if not snapshot.next:  # pragma: no cover
        return None  # pragma: no cover
    ub = _user_interrupt_before(graph)
    ua = _user_interrupt_after(graph)
    if not ub and not ua:
        return None
    ckpt_id = _checkpoint_id(snapshot)
    nxt = tuple(snapshot.next)
    if ua:
        parent_cfg = getattr(snapshot, "parent_config", None)
        if parent_cfg:  # pragma: no cover
            parent = _get_state(graph, parent_cfg)
            prev_ran = [
                t.name
                for t in getattr(parent, "tasks", ()) or ()
                if getattr(t, "result", None) is not None
            ] or list(getattr(parent, "next", ()) or ())
            hit_after = prev_ran if ua == "*" else [n for n in prev_ran if n in ua]
            if hit_after:
                val = {
                    "breakpoint": "interrupt_after",
                    "nodes": hit_after,
                    "next": list(nxt),
                }
                return {
                    "id": f"{STATIC_AFTER_PREFIX}{ckpt_id}:{','.join(hit_after)}",
                    "value": pack(val),
                    "static": True,
                }
    if ub:  # pragma: no cover
        hit_before = list(nxt) if ub == "*" else [n for n in nxt if n in ub]
        if hit_before:
            val = {
                "breakpoint": "interrupt_before",
                "nodes": hit_before,
                "next": list(nxt),
            }
            return {
                "id": f"{STATIC_BEFORE_PREFIX}{ckpt_id}:{','.join(hit_before)}",
                "value": pack(val),
                "static": True,
            }
    return None


def _exec_single_tool(
    graph: CompiledGraph,
    tn: ToolNode | Any,
    snap_values: Any,
    call: Mapping[str, Any],
    thread: str,
    checkpoint_ns: str,
    resume_list: Sequence[Any],
    *,
    handle_tool_errors: bool = False,
    run_config: RunnableConfig | Mapping[str, Any] | None = None,
    context: Any = None,
    node_name: str = "",
) -> tuple[TaskResultStatus, Any]:
    """Execute a single tool call via `ToolNode._run_one` and return `('ok', out)` or.

    `('interrupted', intr)`.

    Normally `ToolNode` executes inside LangGraph's `PregelLoop`, which populates
    `var_child_runnable_config` with a `PregelScratchpad` (`__pregel_scratchpad`),
    channel
    reader (`__pregel_read`), channel writer (`__pregel_send`), and `ToolRuntime`.
    When we extract a `@durable_tool` to run in its own isolated DTS activity
    (`run_tool`),
    there is no surrounding `PregelLoop`. Without a `PregelScratchpad` in the
    contextvar:
      1. Calling `langgraph.types.interrupt()` inside a tool raises `RuntimeError`
         because
         `interrupt()` reads `scratchpad.interrupt_counter` and `scratchpad.resume`.
      2. Tools using `InjectedState`, `InjectedStore`, or returning
         `Command(update=...)`
         fail or lose state updates.
    Synthesizing `PregelScratchpad` (seeded with any previously collected `resume_list`
    values for sequential tool interrupts) and `ToolRuntime` lets `@durable_tool`
    functions
    use the full LangGraph feature set inside an independent DTS activity.
    """
    _, input_type = tn._parse_input(snap_values)
    local_state = copy.deepcopy(snap_values)
    schema = _state_schema_for_graph(graph, node_name)
    state_for_tool = _coerce_with_schema(local_state, schema)
    sent_updates: dict[str, Any] = {}

    def _read(chan: str, fresh: bool = True) -> Any:
        if chan == "__root__":  # pragma: no cover
            return _coerce_with_schema(local_state, schema)  # pragma: no cover
        return (  # pragma: no cover
            local_state.get(chan)
            if isinstance(local_state, dict)
            else getattr(local_state, chan, None)
        )

    def _send(writes: list[tuple[str, Any]]) -> None:
        for ch, val in writes:
            if ch == "__resume__":  # pragma: no cover
                continue
            if isinstance(local_state, dict):  # pragma: no cover
                cur = local_state.get(ch)  # pragma: no cover
                if isinstance(cur, dict) and isinstance(val, dict):  # pragma: no cover
                    merged = dict(cur)  # pragma: no cover
                    for k, v in val.items():  # pragma: no cover
                        if v is None:  # pragma: no cover
                            merged.pop(k, None)  # pragma: no cover
                        else:
                            merged[k] = v  # pragma: no cover
                    local_state[ch] = merged  # pragma: no cover
                else:
                    local_state[ch] = val  # pragma: no cover
            if isinstance(sent_updates.get(ch), dict) and isinstance(
                val, dict
            ):  # pragma: no cover
                sent_updates[ch].update(val)  # pragma: no cover
            else:
                sent_updates[ch] = val  # pragma: no cover

    scratchpad = PregelScratchpad(
        step=0,
        stop=1,
        call_counter=itertools.count(0).__next__,
        interrupt_counter=itertools.count(0).__next__,
        get_null_resume=_null_resume,
        resume=list(resume_list),
        subgraph_counter=itertools.count(0).__next__,
    )
    user_conf = dict(run_config.get("configurable") or {}) if run_config else {}
    run_cfg_dict: dict[str, Any] = {
        "configurable": {
            **user_conf,
            "thread_id": thread,
            "checkpoint_ns": checkpoint_ns
            or f"tool:{call.get('id') or call.get('name')}",
            "__pregel_read": _read,
            "__pregel_send": _send,
            "__pregel_scratchpad": scratchpad,
        },
    }
    if run_config:
        if run_config.get("tags"):  # pragma: no cover
            run_cfg_dict["tags"] = list(run_config["tags"])
        if run_config.get("metadata"):
            run_cfg_dict["metadata"] = dict(run_config["metadata"])
        if run_config.get("recursion_limit") is not None:  # pragma: no cover
            run_cfg_dict["recursion_limit"] = int(
                run_config["recursion_limit"]
            )  # pragma: no cover
    run_cfg = cast(RunnableConfig, run_cfg_dict)
    tr = ToolRuntime(
        state=state_for_tool,
        tool_call_id=call.get("id"),
        config=run_cfg,
        context=context,
        store=getattr(graph, "store", None),
        stream_writer=lambda _: None,
        tools=list(tn.tools_by_name.values()),
        execution_info=None,
        server_info=None,
    )
    run_cfg_dict["configurable"]["__pregel_runtime"] = tr
    tn_exec: Any = tn
    if not handle_tool_errors and getattr(tn, "_handle_tool_errors", False):
        tn_exec = copy.copy(tn)
        tn_exec._handle_tool_errors = False
    tool_obj = getattr(tn_exec, "tools_by_name", {}).get(call.get("name"))
    is_async_only = (
        tool_obj is not None
        and getattr(tool_obj, "func", None) is None
        and getattr(tool_obj, "coroutine", None) is not None
    )

    async def _arun_with_cfg() -> Any:
        tok = var_child_runnable_config.set(run_cfg)
        try:
            return await tn_exec._arun_one(call, input_type, tr)
        finally:
            var_child_runnable_config.reset(tok)

    token = var_child_runnable_config.set(run_cfg)
    try:
        if is_async_only:
            out = _run_async(graph, _arun_with_cfg())
        else:
            try:
                out = tn_exec._run_one(call, input_type, tr)
            except NotImplementedError:
                out = _run_async(graph, _arun_with_cfg())  # pragma: no cover
    except GraphInterrupt as gi:
        intrs = gi.args[0] if gi.args else ()
        intr = intrs[0] if intrs else None
        intr_id = (
            getattr(intr, "id", None) or f"tool-int:{call.get('id')}:{len(resume_list)}"
        )
        intr_val = getattr(intr, "value", None)
        return TaskResultStatus.INTERRUPTED, {"id": intr_id, "value": intr_val}
    finally:
        var_child_runnable_config.reset(token)

    if sent_updates:  # pragma: no cover
        if isinstance(out, Command):  # pragma: no cover
            upd = dict(out.update or {})  # pragma: no cover
            upd.update(sent_updates)  # pragma: no cover
            out = Command(
                graph=out.graph, update=upd, resume=out.resume, goto=out.goto
            )  # pragma: no cover
        else:
            out = Command(
                update={**sent_updates, tn._messages_key: [out]}
            )  # pragma: no cover
    return TaskResultStatus.OK, out


def _get_saved_tuple(
    graph: CompiledGraph, config: RunnableConfig
) -> CheckpointTuple | None:
    """Read the raw `CheckpointTuple` from `graph.checkpointer` (sync or async)."""
    saver = getattr(graph, "checkpointer", None)
    if saver is None:  # pragma: no cover
        return None  # pragma: no cover
    if _is_async_saver(saver):  # pragma: no cover
        return _run_async(graph, saver.aget_tuple(config))  # pragma: no cover
    try:
        return saver.get_tuple(config)
    except (NotImplementedError, asyncio.InvalidStateError):  # pragma: no cover
        _ensure_async_saver(saver)  # pragma: no cover
        return _run_async(graph, saver.aget_tuple(config))  # pragma: no cover


def _prepare_executable_tasks(
    graph: CompiledGraph, snapshot: StateSnapshot | None
) -> Mapping[str, PregelExecutableTask]:
    """Prepare the `dict[task_id, PregelExecutableTask]` for the current `snapshot`.

    checkpoint.
    """
    if snapshot is None or not getattr(snapshot, "config", None):  # pragma: no cover
        return {}  # pragma: no cover
    saved = _get_saved_tuple(graph, snapshot.config)
    if saved is None:  # pragma: no cover
        return {}  # pragma: no cover
    saver = getattr(graph, "checkpointer", None)
    saver_ref = saver if isinstance(saver, BaseCheckpointSaver) else None
    channels, managed = channels_from_checkpoint(
        graph.channels,
        saved.checkpoint,
        saver=saver_ref,
        config=saved.config,
    )
    if null_writes := [
        w[1:] for w in saved.pending_writes or [] if w[0] == NULL_TASK_ID
    ]:  # pragma: no cover
        apply_writes(  # pragma: no cover
            saved.checkpoint,
            channels,
            [PregelTaskWrites((), INPUT, null_writes, [])],
            None,
            graph.trigger_to_nodes,
        )
    step = (saved.metadata or {}).get("step", -1) + 1
    stop = step + 2
    return prepare_next_tasks(
        saved.checkpoint,
        saved.pending_writes or [],
        graph.nodes,
        channels,
        managed,
        saved.config,
        step,
        stop,
        for_execution=True,
        store=getattr(graph, "store", None),
        checkpointer=saver_ref,
        manager=None,
    )


def _exec_single_node(
    graph: CompiledGraph,
    snapshot: StateSnapshot,
    task_id: str,
    node_name: str,
    task_input: Any,
    thread: str,
    resume_list: Sequence[Any],
    *,
    run_config: RunnableConfig | Mapping[str, Any] | None = None,
    context: Any = None,
) -> tuple[TaskResultStatus, Any]:
    """Execute a single `@durable_node` task (`PULL` branch or `Send` `PUSH` task) and.

    return `('ok', out)` or `('interrupted', intr)`.
    """
    exec_tasks = _prepare_executable_tasks(graph, snapshot)
    et = exec_tasks.get(task_id)
    inp_val = (
        et.input
        if et is not None
        else (task_input if task_input is not None else snapshot.values)
    )
    schema = _state_schema_for_graph(graph, node_name)
    if et is None:  # pragma: no cover
        inp_val = _coerce_with_schema(inp_val, schema)  # pragma: no cover

    step = ((getattr(snapshot, "metadata", None) or {}).get("step", -1)) + 1
    scratchpad = PregelScratchpad(
        step=step,
        stop=step + 2,
        call_counter=itertools.count(0).__next__,
        interrupt_counter=itertools.count(0).__next__,
        get_null_resume=_null_resume,
        resume=list(resume_list),
        subgraph_counter=itertools.count(0).__next__,
    )
    if et is not None and isinstance(et.config, Mapping):  # pragma: no cover
        run_cfg_dict: dict[str, Any] = dict(et.config)
        run_cfg_dict[CONF] = dict(run_cfg_dict.get(CONF) or {})
    else:
        run_cfg_dict = dict(_config(thread, run_config=run_config))  # pragma: no cover
        run_cfg_dict[CONF] = dict(run_cfg_dict.get(CONF) or {})  # pragma: no cover

    if run_config:
        user_conf = run_config.get("configurable")
        if isinstance(user_conf, Mapping):  # pragma: no cover
            for k, v in user_conf.items():
                if k not in _INTERNAL_CONF_KEYS and not str(k).startswith(
                    "__pregel_"
                ):  # pragma: no cover
                    run_cfg_dict[CONF][k] = v
        if run_config.get("tags"):
            run_cfg_dict["tags"] = list(run_config["tags"])
        if run_config.get("metadata"):  # pragma: no cover
            merged_meta = dict(run_cfg_dict.get("metadata") or {})  # pragma: no cover
            merged_meta.update(run_config["metadata"])  # pragma: no cover
            run_cfg_dict["metadata"] = merged_meta  # pragma: no cover
        if run_config.get("recursion_limit") is not None:  # pragma: no cover
            run_cfg_dict["recursion_limit"] = int(
                run_config["recursion_limit"]
            )  # pragma: no cover

    rt = run_cfg_dict[CONF].get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)
    rt = rt.override(context=context, store=getattr(graph, "store", None))
    run_cfg_dict[CONF][CONFIG_KEY_RUNTIME] = rt
    run_cfg_dict[CONF][CONFIG_KEY_SCRATCHPAD] = scratchpad
    run_cfg = cast(RunnableConfig, run_cfg_dict)

    pnode = (getattr(graph, "nodes", None) or {}).get(node_name)
    bound = getattr(pnode, "bound", None)
    if bound is None:
        raise RuntimeError(f"unknown graph node {node_name!r}")

    is_async_only = (
        getattr(bound, "func", None) is None
        and getattr(bound, "afunc", None) is not None
    )

    async def _arun_node() -> Any:
        tok = var_child_runnable_config.set(run_cfg)  # pragma: no cover
        try:  # pragma: no cover
            return await bound.ainvoke(inp_val, run_cfg)  # pragma: no cover
        finally:
            var_child_runnable_config.reset(tok)  # pragma: no cover

    token = var_child_runnable_config.set(run_cfg)
    try:
        if is_async_only:  # pragma: no cover
            out = _run_async(graph, _arun_node())  # pragma: no cover
        else:
            try:
                out = bound.invoke(inp_val, run_cfg)
            except (NotImplementedError, TypeError):
                if getattr(bound, "afunc", None) is not None:  # pragma: no cover
                    out = _run_async(graph, _arun_node())  # pragma: no cover
                else:
                    raise  # pragma: no cover
    except GraphInterrupt as gi:
        intrs = gi.args[0] if gi.args else ()
        intr = intrs[0] if intrs else None
        intr_id = getattr(intr, "id", None) or f"node-int:{task_id}:{len(resume_list)}"
        intr_val = getattr(intr, "value", None)
        return TaskResultStatus.INTERRUPTED, {"id": intr_id, "value": intr_val}
    finally:
        var_child_runnable_config.reset(token)

    return TaskResultStatus.OK, out


_REGISTERED_GRAPH_CLONES: dict[
    int, tuple[weakref.ReferenceType[Any], CompiledGraph]
] = {}


def _register_graph_clone(
    orig_graph: CompiledGraph, cloned_graph: CompiledGraph
) -> None:
    """Associate an unmutated caller `orig_graph` with its first registered clone."""
    oid = id(orig_graph)
    existing = _REGISTERED_GRAPH_CLONES.get(oid)
    if existing is not None and existing[0]() is orig_graph:
        return
    try:

        def _on_gc(_r: Any) -> None:
            _REGISTERED_GRAPH_CLONES.pop(oid, None)

        ref = weakref.ref(orig_graph, _on_gc)
        if len(_REGISTERED_GRAPH_CLONES) >= 256:  # pragma: no cover
            _REGISTERED_GRAPH_CLONES.pop(
                next(iter(_REGISTERED_GRAPH_CLONES)), None
            )  # pragma: no cover
        _REGISTERED_GRAPH_CLONES[oid] = (ref, cloned_graph)
    except TypeError:  # pragma: no cover
        pass  # pragma: no cover


def _unwrap_registered_graph(graph: CompiledGraph) -> CompiledGraph:
    """Return the registered clone for `graph` if `graph` is an unmutated caller graph."""  # noqa: E501
    if hasattr(graph, "_durable_name"):
        return graph
    entry = _REGISTERED_GRAPH_CLONES.get(id(graph))
    if entry is not None and entry[0]() is graph:
        return entry[1]
    return graph


def _resolve_graph(graph: CompiledGraph, version: str | None) -> CompiledGraph:
    """Resolve the compiled `graph` for `version` (returning a previous version if.

    registered, or raising `RunError`).
    """
    graph = _unwrap_registered_graph(graph)
    if version is None:
        return graph
    cur_ver = getattr(graph, "_durable_version", None)
    if cur_ver is not None and version == cur_ver:
        return graph
    prev = getattr(graph, "_durable_previous_versions", None) or {}
    if version in prev:
        return prev[version]
    gname = getattr(graph, "_durable_name", None) or getattr(graph, "name", "graph")
    raise RunError(
        f"unknown graph version {version!r} for graph {gname!r} "
        f"(registered: current={cur_ver!r}, previous={sorted(prev)})"
    )


def _stamp_migrated_metadata(
    graph: CompiledGraph, snap: StateSnapshot | None, to_version: str | None
) -> None:
    """Record `migrated_to` on the current checkpoint's metadata so redelivered.

    migrations are idempotent.
    """
    saver: Any = getattr(graph, "checkpointer", None)
    if (
        saver is None or snap is None or not getattr(snap, "config", None)
    ):  # pragma: no cover
        return  # pragma: no cover
    saved = _get_saved_tuple(graph, snap.config)
    if saved is None:  # pragma: no cover
        return  # pragma: no cover
    p_cfg_dict: dict[str, Any] = dict(saved.config)
    p_cfg_dict["configurable"] = dict(p_cfg_dict.get("configurable") or {})
    if saved.parent_config and (saved.parent_config.get("configurable") or {}).get(
        "checkpoint_id"
    ):  # pragma: no cover
        p_cfg_dict["configurable"]["checkpoint_id"] = saved.parent_config[
            "configurable"
        ]["checkpoint_id"]
    else:
        p_cfg_dict["configurable"].pop("checkpoint_id", None)  # pragma: no cover
    p_cfg = cast(RunnableConfig, p_cfg_dict)
    new_meta: Any = {
        **(saved.metadata or {}),
        "migrated_to": str(to_version or "migrated"),
    }
    if _is_async_saver(saver):  # pragma: no cover
        _run_async(
            graph, saver.aput(p_cfg, saved.checkpoint, new_meta, {})
        )  # pragma: no cover
    else:
        try:
            saver.put(p_cfg, saved.checkpoint, new_meta, {})
        except (NotImplementedError, asyncio.InvalidStateError):  # pragma: no cover
            _ensure_async_saver(saver)  # pragma: no cover
            _run_async(
                graph, saver.aput(p_cfg, saved.checkpoint, new_meta, {})
            )  # pragma: no cover


def _apply_version_migration(
    root_graph: CompiledGraph,
    target_graph: CompiledGraph,
    thread: str,
    migrate_from: str | None,
    to_version: str | None,
    *,
    request: Mapping[str, Any] | None = None,
    run_config: RunnableConfig | Mapping[str, Any] | None = None,
) -> bool:
    """Apply a registered or explicit version migration to `thread`'s checkpoint on.

    `target_graph`.

    Returns `True` if `_update_state(..., as_node=...)` was applied (meaning the
    checkpoint jumped
    to the outgoing edges of `as_node`), or `False` if state channels were updated
    in-place via `NULL_TASK_ID`.
    """
    req = request or {}
    config = _config(thread, run_config=run_config)
    old_graph = _resolve_graph(root_graph, migrate_from) if migrate_from else root_graph
    snap = _get_state(old_graph, config)

    old_saver = getattr(old_graph, "checkpointer", None)
    target_saver: Any = getattr(target_graph, "checkpointer", None)
    if (
        old_saver is not None
        and target_saver is not None
        and old_saver is not target_saver
    ):  # pragma: no cover
        bundle = _export_checkpoint_bundle(old_graph, thread)  # pragma: no cover
        if bundle is not None:  # pragma: no cover
            _hydrate_checkpoint_bundle(target_graph, thread, bundle)  # pragma: no cover
        else:
            saved = _get_saved_tuple(old_graph, config)  # pragma: no cover
            if (
                saved is not None
                and _get_saved_tuple(target_graph, saved.config) is None
            ):  # pragma: no cover
                p_cfg = saved.parent_config or config  # pragma: no cover
                if _is_async_saver(target_saver):  # pragma: no cover
                    _run_async(  # pragma: no cover
                        target_graph,
                        target_saver.aput(
                            p_cfg, saved.checkpoint, saved.metadata or {}, {}
                        ),
                    )
                else:
                    target_saver.put(
                        p_cfg, saved.checkpoint, saved.metadata or {}, {}
                    )  # pragma: no cover

    before_values = (
        copy.deepcopy(snap.values) if isinstance(snap.values, dict) else snap.values
    )
    cur_values = (
        copy.deepcopy(before_values)
        if isinstance(before_values, dict)
        else before_values
    )

    explicit_as_node = req.get("as_node")
    explicit_values = unpack(req["migrate_values"]) if "migrate_values" in req else None
    resume_val = unpack(req["resume"]) if "resume" in req else None

    meta = getattr(snap, "metadata", None) or {}
    target_tag = str(to_version or "migrated")
    if (
        meta.get("migrated_to") == target_tag
        and explicit_as_node is None
        and explicit_values is None
    ):  # pragma: no cover
        return False  # pragma: no cover

    migrations = getattr(root_graph, "_durable_migrations", None) or {}
    mig = None
    for key in (
        (migrate_from, to_version),
        f"{migrate_from}->{to_version}",
        to_version,
    ):
        if key in migrations:
            mig = migrations[key]
            break

    mig_out: Any = None
    if callable(mig):
        try:
            sig = inspect.signature(mig)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            has_varargs = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL
                for p in sig.parameters.values()
            )
            if has_varargs or len(params) >= 3:
                mig_out = mig(cur_values, snap, resume_val)
            elif len(params) == 2:  # pragma: no cover
                mig_out = mig(cur_values, snap)  # pragma: no cover
            else:
                mig_out = mig(cur_values)
        except TypeError:  # pragma: no cover
            mig_out = mig(cur_values)  # pragma: no cover
    elif isinstance(mig, Mapping):  # pragma: no cover
        mig_out = dict(mig)  # pragma: no cover

    mig_values: Any = None
    mig_as_node: str | None = explicit_as_node
    if isinstance(mig_out, Mapping):
        if "as_node" in mig_out or "values" in mig_out or "node_map" in mig_out:
            if "as_node" in mig_out and not mig_as_node:  # pragma: no cover
                mig_as_node = mig_out["as_node"]
            if (
                "node_map" in mig_out
                and isinstance(mig_out["node_map"], Mapping)
                and not mig_as_node
            ):  # pragma: no cover
                nxt = getattr(snap, "next", ()) or ()  # pragma: no cover
                if nxt and nxt[0] in mig_out["node_map"]:  # pragma: no cover
                    mig_as_node = mig_out["node_map"][nxt[0]]  # pragma: no cover
            if "values" in mig_out:  # pragma: no cover
                mig_values = mig_out["values"]
                if callable(mig_values):  # pragma: no cover
                    mig_values = mig_values(cur_values)  # pragma: no cover
        else:
            mig_values = dict(mig_out)

    if explicit_values is not None:  # pragma: no cover
        if isinstance(mig_values, dict) and isinstance(
            explicit_values, Mapping
        ):  # pragma: no cover
            mig_values = {**mig_values, **explicit_values}  # pragma: no cover
        else:
            mig_values = explicit_values  # pragma: no cover

    delta: dict[str, Any] = {}
    if isinstance(mig_values, Mapping):
        channels = getattr(target_graph, "channels", None) or {}
        before_map = before_values if isinstance(before_values, Mapping) else {}
        for k, v in mig_values.items():
            if k in before_map and v == before_map[k]:  # pragma: no cover
                continue  # pragma: no cover
            ch_obj = channels.get(k)
            old_v = before_map.get(k)
            if (
                ch_obj is not None
                and hasattr(ch_obj, "operator")
                and not isinstance(v, Overwrite)
            ):
                if (
                    isinstance(old_v, list)
                    and isinstance(v, list)
                    and len(old_v) > 0
                    and len(v) > len(old_v)
                    and v[: len(old_v)] == old_v
                ):
                    v = v[len(old_v) :]
                else:
                    v = Overwrite(v)
            delta[k] = v

    if mig_as_node is not None:
        _update_state(target_graph, config, delta, as_node=mig_as_node)
        return True

    if delta and target_saver is not None:
        writes = list(delta.items())
        if _is_async_saver(target_saver):  # pragma: no cover
            _run_async(  # pragma: no cover
                target_graph,
                target_saver.aput_writes(snap.config, writes, NULL_TASK_ID),
            )
        else:
            try:
                target_saver.put_writes(snap.config, writes, NULL_TASK_ID)
            except (NotImplementedError, asyncio.InvalidStateError):  # pragma: no cover
                _ensure_async_saver(target_saver)  # pragma: no cover
                _run_async(  # pragma: no cover
                    target_graph,
                    target_saver.aput_writes(snap.config, writes, NULL_TASK_ID),
                )
    _stamp_migrated_metadata(target_graph, snap, to_version)
    return False
