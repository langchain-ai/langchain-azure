# Copyright (c) Microsoft. All rights reserved.

"""Core execution engine: `_report`, `run_tool`, `run_node`, `run_step`, and.

`make_orchestrator`.

This module coordinates the four Durable Task primitives per attached graph
(`langgraph.<name>.{run,step,tool,node}`):
  1. `make_orchestrator(limits)` (`langgraph.<name>.run`): deterministic generator
     orchestration driving the
     superstep loop, per-tool and per-node durable activities, `interrupt()` external
     events, and `continue_as_new`.
  2. `run_step(graph, limits, request)` (`langgraph.<name>.step`): idempotent superstep
     activity guarded by
     composite `expect` checkpoint matching and `_is_mid_start` crash detection.
  3. `run_tool(graph, request)` (`langgraph.<name>.tool`): isolated single-tool activity
     with a `PregelScratchpad`.
  4. `run_node(graph, request)` (`langgraph.<name>.node`): isolated `@durable_node`
     activity for parallel branches and `Send()`.

All helper symbols from `_config`, `_async_bridge`, `_inline_saver`, and `_checkpoints`
are re-exported here for internal orchestration and activity execution.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

from durabletask import task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, StateSnapshot

from langchain_azure_compute.graph._async_bridge import (
    _BG_LOCK,
    _BG_LOOP,
    _BG_THREAD,
    _ensure_async_saver,
    _get_bg_loop,
    _is_async_saver,
    _run_async,
)
from langchain_azure_compute.graph._checkpoints import (
    _DURABLE_ERROR_MARKER,
    _apply_version_migration,
    _checkpoint_id,
    _checkpoint_match,
    _coerce_with_schema,
    _config,
    _durable_node_names,
    _durable_tool_node_names,
    _exec_single_node,
    _exec_single_tool,
    _find_subgraph_tool_target,
    _format_failed_tool_message,
    _get_state,
    _invoke,
    _is_mid_start,
    _merged_interrupt_before,
    _normalize_run_config,
    _pack_context,
    _parent_checkpoint_id,
    _pending_tool_tasks,
    _prepare_executable_tasks,
    _prepare_subgraph_tool_nodes,
    _project_output,
    _put_task_writes,
    _raw_checkpoint_id,
    _resolve_graph,
    _state_schema_for_graph,
    _static_breakpoint_hit,
    _tool_node_handles_errors,
    _tool_nodes,
    _tool_task_graph,
    _unpack_context,
    _update_state,
    _user_interrupt_after,
    _user_interrupt_before,
    read_output,
)
from langchain_azure_compute.graph._config import (
    _SERDE,
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
    _check_payload,
    _effective_node_config,
    _effective_tool_config,
    _parse_policy_value,
    _selector_key,
    _size,
    _validate_graph_name,
    durable_node,
    durable_tool,
    mark_node,
    mark_tool,
    pack,
    unpack,
)
from langchain_azure_compute.graph._inline_saver import (
    _INLINE_SAVER_LOCK,
    _INLINE_THREAD_BUNDLES,
    MAX_INLINE_THREAD_BUNDLES,
    _decode_typed,
    _encode_typed,
    _ensure_checkpointer,
    _export_checkpoint_bundle,
    _hydrate_checkpoint_bundle,
    _is_inline_checkpointer,
)

# Compatibility re-exports consumed by the API facade and existing integrations.
__all__ = [
    "run_step",
    "run_tool",
    "run_node",
    "make_orchestrator",
    "_BG_LOCK",
    "_BG_LOOP",
    "_BG_THREAD",
    "_ensure_async_saver",
    "_get_bg_loop",
    "_is_async_saver",
    "_run_async",
    "_DURABLE_ERROR_MARKER",
    "_apply_version_migration",
    "_checkpoint_id",
    "_checkpoint_match",
    "_coerce_with_schema",
    "_config",
    "_durable_node_names",
    "_durable_tool_node_names",
    "_exec_single_node",
    "_exec_single_tool",
    "_find_subgraph_tool_target",
    "_format_failed_tool_message",
    "_get_state",
    "_invoke",
    "_is_mid_start",
    "_merged_interrupt_before",
    "_normalize_run_config",
    "_pack_context",
    "_parent_checkpoint_id",
    "_pending_tool_tasks",
    "_prepare_executable_tasks",
    "_prepare_subgraph_tool_nodes",
    "_project_output",
    "_put_task_writes",
    "_raw_checkpoint_id",
    "_resolve_graph",
    "_state_schema_for_graph",
    "_static_breakpoint_hit",
    "_tool_node_handles_errors",
    "_tool_nodes",
    "_tool_task_graph",
    "_unpack_context",
    "_update_state",
    "_user_interrupt_after",
    "_user_interrupt_before",
    "read_output",
    "_SERDE",
    "CUSTOM_STATUS_INLINE_LIMIT",
    "DURABLE_TOOL_ERROR_KEY",
    "MAX_GRAPH_NAME_LENGTH",
    "RESUME_EVENT_PREFIX",
    "STATIC_AFTER_PREFIX",
    "STATIC_BEFORE_PREFIX",
    "STATIC_INTERRUPT_PREFIX",
    "CheckpointBundle",
    "Checkpointer",
    "ClientProtocol",
    "CompiledGraph",
    "ExecutionMode",
    "InterruptEntry",
    "Limits",
    "MigrationKey",
    "MigrationSpec",
    "NodeCallable",
    "NodeConfig",
    "NodeReport",
    "NodeRequest",
    "NodeTaskSpec",
    "OrchestratorInput",
    "OrchestratorOutput",
    "OrchestratorState",
    "PackedBlob",
    "PolicyValue",
    "RunError",
    "Selector",
    "StepKind",
    "StepReport",
    "StepRequest",
    "StepStatus",
    "TaskResultStatus",
    "ToolCallable",
    "ToolCallSpec",
    "ToolConfig",
    "ToolReport",
    "ToolRequest",
    "WorkerProtocol",
    "_check_payload",
    "_effective_node_config",
    "_effective_tool_config",
    "_parse_policy_value",
    "_selector_key",
    "_size",
    "_validate_graph_name",
    "durable_node",
    "durable_tool",
    "mark_node",
    "mark_tool",
    "pack",
    "unpack",
    "_INLINE_SAVER_LOCK",
    "_INLINE_THREAD_BUNDLES",
    "MAX_INLINE_THREAD_BUNDLES",
    "_decode_typed",
    "_encode_typed",
    "_ensure_checkpointer",
    "_export_checkpoint_bundle",
    "_hydrate_checkpoint_bundle",
    "_is_inline_checkpointer",
]


def _checked_report(report: dict[str, Any]) -> dict[str, Any]:
    _check_payload(report, "activity result")
    return report


def _remember(saver: Any, attribute: str, key: tuple[Any, ...]) -> None:
    cache = getattr(saver, attribute, None)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict((item, None) for item in (cache or ()))
        setattr(saver, attribute, cache)
    cache[key] = None
    cache.move_to_end(key)
    while len(cache) > 256:
        cache.popitem(last=False)


def _start_lineage_matches(
    graph: CompiledGraph,
    snapshot: StateSnapshot,
    expect: str | None,
    instance_id: str,
    input_sig: str,
) -> bool:
    current = snapshot
    while _raw_checkpoint_id(current) != _raw_checkpoint_id(expect):
        metadata = dict(current.metadata or {})
        if "durable_instance_id" in metadata:
            return (
                metadata["durable_instance_id"] == instance_id
                and metadata.get("durable_input_sig") == input_sig
                and metadata.get("durable_expect") == (expect or "")
            )
        if not current.parent_config:
            break
        current = _get_state(graph, current.parent_config)
    return False


def _report(
    graph: CompiledGraph,
    snapshot: StateSnapshot,
    limits: Limits,
    *,
    replayed: bool = False,
    tool_config: ToolConfig | None = None,
    node_config: NodeConfig | None = None,
    max_concurrency: int | None = None,
    node_max_concurrency: int | None = None,
    version: str | None = None,
    root_graph: CompiledGraph | None = None,
) -> dict[str, Any]:
    """Inspect post-step `snapshot` and build the control envelope (`interrupted | done.

    | tools | nodes | running`).
    """
    interrupts = [i for t in snapshot.tasks for i in t.interrupts]
    ckpt_id = _checkpoint_id(snapshot)
    report: dict[str, Any] = {"ckpt": ckpt_id, "replayed": replayed}
    if root_graph is not None and getattr(root_graph, "_durable_upgrade_on_can", False):
        latest_ver = getattr(root_graph, "_durable_version", None)
        if latest_ver is not None and version is not None and latest_ver != version:
            report["target_version"] = latest_ver
    thread = (snapshot.config or {}).get("configurable", {}).get("thread_id")
    if thread and _is_inline_checkpointer(graph):
        bundle = _export_checkpoint_bundle(graph, thread)
        if bundle is not None:
            report["checkpoint_bundle"] = bundle
    if interrupts:
        items: list[dict[str, Any]] = []
        for intr in interrupts:
            packed = pack(intr.value)
            entry: dict[str, Any] = {"id": intr.id}
            if _size(packed) <= limits.custom_status_limit:
                entry["value"] = packed
            else:
                entry["in_checkpoint"] = True
            items.append(entry)
        report.update(
            status=StepStatus.INTERRUPTED, interrupt=items[0], interrupts=items
        )
        return _checked_report(report)

    if not snapshot.next:
        report["status"] = StepStatus.DONE
        if _is_inline_checkpointer(graph) and not report.get("checkpoint_bundle"):
            report["output"] = pack(_project_output(graph, snapshot.values))
        return _checked_report(report)

    tcfg = _effective_tool_config(graph, tool_config)
    tnodes = _tool_nodes(graph)
    target_node: str | None = None
    target_snap: StateSnapshot = snapshot
    target_ns, raw_target_ckpt = "", _raw_checkpoint_id(snapshot) or ""

    tasks = getattr(snapshot, "tasks", None) or ()
    uncompleted_nodes = {t.name for t in tasks if getattr(t, "result", None) is None}
    completed_tasks = {
        t.name
        for t in tasks
        if getattr(t, "result", None) is not None and t.name not in uncompleted_nodes
    }
    top_matches = [n for n in snapshot.next if n in tnodes and n not in completed_tasks]
    if top_matches:
        target_node = top_matches[0]
    else:
        sub_target = _find_subgraph_tool_target(snapshot, tnodes)
        if sub_target is not None:
            target_node, _, target_snap, ns_and_ckpt = sub_target
            target_ns, raw_target_ckpt = ns_and_ckpt.split("|", 1)

    if target_node is not None:
        tn = tnodes[target_node]
        pending_tasks = _pending_tool_tasks(graph, target_snap, target_node, tn)
        tool_calls = [
            (task_id, call, input_type)
            for task_id, calls, input_type in pending_tasks
            for call in calls
        ]
        if pending_tasks:
            handle_errors = _tool_node_handles_errors(tn)
            specs = []
            for idx, (task_id, call, input_type) in enumerate(tool_calls):
                is_dur, attempts = tcfg.resolve(tn, target_node, call["name"], limits)
                call_id = f"{ckpt_id}:{target_node}:{idx}:{call.get('id') or idx}"
                specs.append(
                    {
                        "call_id": call_id,
                        "task_id": task_id,
                        "input_type": input_type,
                        "index": idx,
                        "id": call.get("id"),
                        "name": call["name"],
                        "call": pack(call),
                        "durable": is_dur,
                        "retries": attempts,
                        "handle_errors": handle_errors,
                    }
                )
            if not specs or any(s["durable"] for s in specs):
                eff_mc = (
                    int(max_concurrency)
                    if max_concurrency is not None
                    else tcfg.max_concurrency
                )
                report.update(
                    status=StepStatus.TOOLS,
                    node=target_node,
                    tool_calls=specs,
                    tool_ns=target_ns,
                    tool_ckpt=raw_target_ckpt,
                )
                if eff_mc is not None:
                    report["max_concurrency"] = eff_mc
                return _checked_report(report)

    static_intr = _static_breakpoint_hit(graph, snapshot)
    if static_intr is not None:
        report.update(
            status=StepStatus.INTERRUPTED,
            interrupt=static_intr,
            interrupts=[static_intr],
        )
        return _checked_report(report)

    ncfg = _effective_node_config(graph, node_config)
    uncompleted_tasks = [t for t in tasks if getattr(t, "result", None) is None]
    if uncompleted_tasks:
        node_decisions = [
            ncfg.resolve(graph, t.name, limits) for t in uncompleted_tasks
        ]
        if any(is_dur for is_dur, _ in node_decisions):
            has_send = any(
                bool(t_path and t_path[0] == "__pregel_push")
                for t in uncompleted_tasks
                if (t_path := getattr(t, "path", None))
            )
            exec_tasks = _prepare_executable_tasks(graph, snapshot) if has_send else {}
            node_specs = []
            for idx, (t, (is_dur, attempts)) in enumerate(
                zip(uncompleted_tasks, node_decisions)
            ):
                t_path = getattr(t, "path", None)
                is_send = bool(t_path and t_path[0] == "__pregel_push")
                call_id = f"{ckpt_id}:node:{t.name}:{idx}:{t.id}"
                spec: dict[str, Any] = {
                    "call_id": call_id,
                    "index": idx,
                    "task_id": t.id,
                    "name": t.name,
                    "durable": is_dur,
                    "retries": attempts,
                }
                if is_send:
                    et = exec_tasks.get(t.id)
                    task_inp = et.input if et is not None else snapshot.values
                    spec["input"] = pack(task_inp)
                node_specs.append(spec)
            eff_nmc = (
                int(node_max_concurrency)
                if node_max_concurrency is not None
                else (
                    int(max_concurrency)
                    if max_concurrency is not None
                    else ncfg.max_concurrency
                )
            )
            report.update(status=StepStatus.NODES, node_tasks=node_specs)
            if eff_nmc is not None:
                report["max_concurrency"] = eff_nmc
            return _checked_report(report)

    report["status"] = StepStatus.RUNNING
    return _checked_report(report)


def run_tool(
    graph: CompiledGraph, request: ToolRequest | Mapping[str, Any]
) -> dict[str, Any]:
    """Execute a single `@durable_tool` call inside its own DTS activity.

    (`tool_activity`).
    """
    _check_payload(request, "tool activity input")
    graph = _resolve_graph(graph, request.get("version"))
    saver = _ensure_checkpointer(graph)
    thread, ckpt_id = request["thread"], _raw_checkpoint_id(request["ckpt"])
    node_name, tool_ns = request["node"], request.get("tool_ns", "")
    tool_ckpt = (
        request.get("tool_ckpt") if tool_ns else (request.get("tool_ckpt") or ckpt_id)
    )
    call = unpack(request["call"])
    resumes = [unpack(r) for r in request.get("resumes", ())]
    run_config = (
        unpack(request["run_config"])
        if "run_config" in request
        else _normalize_run_config(request.get("config"))
    )
    context = (
        _unpack_context(graph, request["context"]) if "context" in request else None
    )

    run_config = dict(run_config or {})
    run_config["metadata"] = {
        **run_config.get("metadata", {}),
        "durable_call_id": request["call_id"],
    }
    tn = _tool_nodes(graph).get(node_name)
    if tn is None:
        raise RunError(f"unknown tool node {node_name!r}")

    target_cfg = _config(thread, tool_ckpt, tool_ns, run_config=run_config)
    bundle = request.get("checkpoint_bundle")
    if bundle is not None and isinstance(saver, InMemorySaver):
        _hydrate_checkpoint_bundle(
            graph, thread, bundle, only_if_missing_cfg=target_cfg
        )
    snap = _get_state(graph, target_cfg)
    call_key = call.get("id") or request.get("index", 0)
    eff_tool_ns = f"{tool_ns}|{call_key}" if tool_ns else f"{node_name}:{call_key}"
    status, res = _exec_single_tool(
        graph,
        tn,
        snap.values,
        call,
        thread,
        eff_tool_ns,
        resumes,
        handle_tool_errors=False,
        run_config=run_config,
        context=context,
        node_name=node_name,
        task_input_type=request.get("input_type"),
    )
    if status == StepStatus.INTERRUPTED:
        return _checked_report(
            {
                "call_id": request["call_id"],
                "status": StepStatus.INTERRUPTED,
                "interrupt": {"id": res["id"], "value": pack(res["value"])},
            }
        )
    return _checked_report(
        {"call_id": request["call_id"], "status": StepStatus.OK, "output": pack(res)}
    )


def run_node(
    graph: CompiledGraph, request: NodeRequest | Mapping[str, Any]
) -> dict[str, Any]:
    """Execute a single `@durable_node` task (`PULL` branch or `Send` `PUSH` task).

    inside its own DTS activity (`node_activity`).
    """
    _check_payload(request, "node activity input")
    graph = _resolve_graph(graph, request.get("version"))
    saver = _ensure_checkpointer(graph)
    thread, ckpt_id = request["thread"], _raw_checkpoint_id(request["ckpt"])
    node_name, task_id = request["name"], request["task_id"]
    task_inp = (
        unpack(request["input"])
        if "input" in request and request["input"] is not None
        else None
    )
    resumes = [unpack(r) for r in request.get("resumes", ())]
    run_config = (
        unpack(request["run_config"])
        if "run_config" in request
        else _normalize_run_config(request.get("config"))
    )
    context = (
        _unpack_context(graph, request["context"]) if "context" in request else None
    )

    target_cfg = _config(thread, ckpt_id, "", run_config=run_config)
    bundle = request.get("checkpoint_bundle")
    if bundle is not None and isinstance(saver, InMemorySaver):
        _hydrate_checkpoint_bundle(
            graph, thread, bundle, only_if_missing_cfg=target_cfg
        )
    snap = _get_state(graph, target_cfg)
    status, res = _exec_single_node(
        graph,
        snap,
        task_id,
        node_name,
        task_inp,
        thread,
        resumes,
        run_config=run_config,
        context=context,
    )
    if status == StepStatus.INTERRUPTED:
        return _checked_report(
            {
                "call_id": request["call_id"],
                "status": StepStatus.INTERRUPTED,
                "interrupt": {"id": res["id"], "value": pack(res["value"])},
            }
        )
    return _checked_report(
        {"call_id": request["call_id"], "status": StepStatus.OK, "output": pack(res)}
    )


def run_step(
    graph: CompiledGraph,
    limits: Limits,
    request: StepRequest | Mapping[str, Any],
    *,
    tool_config: ToolConfig | None = None,
    node_config: NodeConfig | None = None,
) -> dict[str, Any]:
    """Advance one LangGraph superstep or commit tool and node outputs.

    Re-delivery is safe: if the thread already moved past expect, return a replayed
    report without running the step again.
    """
    from langchain_azure_compute.graph._checkpoints import _unwrap_registered_graph

    _check_payload(request, "step activity input")
    root_graph = _unwrap_registered_graph(graph)
    req_version = request.get("version")
    migrate_from = request.get("migrate_from")
    graph = _resolve_graph(root_graph, req_version)
    saver = _ensure_checkpointer(graph)
    thread = request["thread"]
    run_config = (
        unpack(request["run_config"])
        if "run_config" in request
        else _normalize_run_config(request.get("config"))
    )
    context = (
        _unpack_context(graph, request["context"]) if "context" in request else None
    )
    config = _config(thread, run_config=run_config)
    tcfg = _effective_tool_config(graph, tool_config)
    ncfg = _effective_node_config(graph, node_config)
    req_mc = request.get("max_concurrency")
    eff_mc = int(req_mc) if req_mc is not None else tcfg.max_concurrency
    req_nmc = request.get("node_max_concurrency")
    eff_nmc = (
        int(req_nmc)
        if req_nmc is not None
        else (int(req_mc) if req_mc is not None else ncfg.max_concurrency)
    )
    _prepare_subgraph_tool_nodes(graph, limits, tcfg)
    check_graph = (
        _resolve_graph(root_graph, migrate_from)
        if (migrate_from is not None and migrate_from != req_version)
        else graph
    )
    has_subgraphs = bool(
        callable(getattr(graph, "get_subgraphs", None)) and list(graph.get_subgraphs())
    )
    before = _get_state(check_graph, config, subgraphs=has_subgraphs)
    expect, kind = request.get("expect"), request["kind"]
    durable_tnodes = _durable_tool_node_names(graph, limits, tcfg, before)
    durable_nnodes = _durable_node_names(graph, limits, ncfg, before)
    durable_pause_nodes = [
        *durable_tnodes,
        *[n for n in durable_nnodes if n not in durable_tnodes],
    ]
    bundle = request.get("checkpoint_bundle")
    if (
        bundle is not None
        and isinstance(saver, InMemorySaver)
        and not _checkpoint_match(before, expect)
    ):
        completed_expects = getattr(saver, "_durable_completed_expects", None) or set()
        ib_check = _merged_interrupt_before(check_graph, durable_pause_nodes)
        already_ran_here = (thread, expect) in completed_expects or (
            kind == StepKind.START
            and _is_mid_start(check_graph, before, expect, ib_check)
        )
        if not already_ran_here:
            _hydrate_checkpoint_bundle(check_graph, thread, bundle, preserve_newer=True)
            before = _get_state(check_graph, config, subgraphs=has_subgraphs)
            durable_tnodes = _durable_tool_node_names(graph, limits, tcfg, before)
            durable_nnodes = _durable_node_names(graph, limits, ncfg, before)
            durable_pause_nodes = [
                *durable_tnodes,
                *[n for n in durable_nnodes if n not in durable_tnodes],
            ]

    before_meta = getattr(before, "metadata", None) or {}
    before_step = before_meta.get("step", -2) if before_meta else -2
    owner = request.get("owner")
    checkpoint_owner = before_meta.get("durable_owner")
    if checkpoint_owner is not None and checkpoint_owner != owner:
        raise PermissionError("Checkpoint thread belongs to another owner")
    if owner is not None:
        config["metadata"] = {**config.get("metadata", {}), "durable_owner": owner}

    if kind == StepKind.START:
        ib = _merged_interrupt_before(graph, durable_pause_nodes)
        inp_val = unpack(request["input"])
        inp_sig = hashlib.sha256(
            json.dumps(request.get("input"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        instance_id = (
            request.get("instance_id")
            or hashlib.sha256(
                json.dumps([thread, expect, inp_sig]).encode("utf-8")
            ).hexdigest()
        )
        if not _checkpoint_match(before, expect) and not _start_lineage_matches(
            graph, before, expect, instance_id, inp_sig
        ):
            raise RunError(
                f"Concurrent turn conflict on thread {thread!r}: checkpoint moved "
                f"from {expect!r} to {_checkpoint_id(before)!r} by another run"
            )
        config["metadata"] = {
            **config.get("metadata", {}),
            "durable_instance_id": instance_id,
            "durable_input_sig": inp_sig,
            "durable_expect": expect or "",
        }
        if _checkpoint_match(before, expect):
            actual_input = inp_val
        elif _is_mid_start(graph, before, expect, ib):
            actual_input = None
        else:
            return _report(
                graph,
                before,
                limits,
                replayed=True,
                tool_config=tcfg,
                node_config=ncfg,
                max_concurrency=eff_mc,
                node_max_concurrency=eff_nmc,
                version=req_version,
                root_graph=root_graph,
            )
        _invoke(
            graph,
            actual_input,
            config,
            interrupt_after="*",
            interrupt_before=ib,
            max_concurrency=eff_mc,
            context=context,
            before_step=before_step,
        )
        _remember(saver, "_durable_completed_starts", (thread, expect, inp_sig))
    else:
        mig_requested = (
            (migrate_from is not None and migrate_from != req_version)
            or kind == StepKind.MIGRATE
            or "as_node" in request
            or "migrate_values" in request
        )
        mid_mig_resume = False
        mid_mig_jumped = False
        if not _checkpoint_match(before, expect):
            after_snap = (
                _get_state(graph, config, subgraphs=has_subgraphs)
                if check_graph is not graph
                and _checkpoint_id(_get_state(graph, config, subgraphs=has_subgraphs))
                is not None
                else before
            )
            if kind == StepKind.RESUME and mig_requested and expect is not None:
                a_meta = getattr(after_snap, "metadata", None) or {}
                raw_exp = _raw_checkpoint_id(expect)
                target_tag = str(req_version or "migrated")
                if (
                    a_meta.get("migrated_to") == target_tag
                    and _raw_checkpoint_id(after_snap) == raw_exp
                ):
                    mid_mig_resume = True
                    before = after_snap
                elif (
                    a_meta.get("source") == "update"
                    and _parent_checkpoint_id(after_snap) == raw_exp
                ):
                    mid_mig_resume = True
                    mid_mig_jumped = True
                    before = after_snap
            if not mid_mig_resume:
                return _report(
                    graph,
                    after_snap,
                    limits,
                    replayed=True,
                    tool_config=tcfg,
                    node_config=ncfg,
                    max_concurrency=eff_mc,
                    node_max_concurrency=eff_nmc,
                    version=req_version,
                    root_graph=root_graph,
                )
            durable_tnodes = _durable_tool_node_names(graph, limits, tcfg, before)
            durable_nnodes = _durable_node_names(graph, limits, ncfg, before)
            durable_pause_nodes = [
                *durable_tnodes,
                *[n for n in durable_nnodes if n not in durable_tnodes],
            ]
            before_meta = getattr(before, "metadata", None) or {}
            before_step = before_meta.get("step", -2) if before_meta else -2
        jumped = mid_mig_jumped
        if mid_mig_resume:
            if jumped:
                kind = StepKind.STEP
        elif migrate_from is not None and migrate_from != req_version:
            jumped = _apply_version_migration(
                root_graph,
                graph,
                thread,
                migrate_from,
                req_version,
                request=request,
                run_config=run_config,
            )
            before = _get_state(graph, config, subgraphs=has_subgraphs)
            durable_tnodes = _durable_tool_node_names(graph, limits, tcfg, before)
            durable_nnodes = _durable_node_names(graph, limits, ncfg, before)
            durable_pause_nodes = [
                *durable_tnodes,
                *[n for n in durable_nnodes if n not in durable_tnodes],
            ]
            if jumped and kind == StepKind.RESUME:
                kind = StepKind.STEP
        elif (
            kind == StepKind.MIGRATE
            or "as_node" in request
            or "migrate_values" in request
        ):
            jumped = _apply_version_migration(
                root_graph,
                graph,
                thread,
                migrate_from or req_version,
                req_version,
                request=request,
                run_config=run_config,
            )
            before = _get_state(graph, config, subgraphs=has_subgraphs)
            durable_tnodes = _durable_tool_node_names(graph, limits, tcfg, before)
            durable_nnodes = _durable_node_names(graph, limits, ncfg, before)
            durable_pause_nodes = [
                *durable_tnodes,
                *[n for n in durable_nnodes if n not in durable_tnodes],
            ]
            if jumped and kind == StepKind.RESUME:
                kind = StepKind.STEP

        if kind in (StepKind.MIGRATE, StepKind.INSPECT):
            if (kind == StepKind.MIGRATE or migrate_from is not None) and isinstance(
                saver, InMemorySaver
            ):
                _remember(saver, "_durable_completed_expects", (thread, expect))
            rep = _report(
                graph,
                before,
                limits,
                tool_config=tcfg,
                node_config=ncfg,
                max_concurrency=eff_mc,
                node_max_concurrency=eff_nmc,
                version=req_version,
                root_graph=root_graph,
            )
            if kind == StepKind.MIGRATE:
                rep["jumped"] = jumped
            return rep
        elif kind == StepKind.RESUME:
            interrupts = [i for t in before.tasks for i in t.interrupts]
            if not interrupts and before.next:
                ib = _merged_interrupt_before(
                    graph, durable_pause_nodes, exclude_static=list(before.next)
                )
                _invoke(
                    graph,
                    None,
                    config,
                    interrupt_after="*",
                    interrupt_before=ib,
                    max_concurrency=eff_mc,
                    context=context,
                    before_step=before_step,
                )
            else:
                res_val: Any
                raw_resume = request["resume"]
                raw_map = (
                    raw_resume.get("__resume_map__")
                    if isinstance(raw_resume, Mapping)
                    else None
                )
                if isinstance(raw_map, Mapping):
                    res_val = {
                        str(k): (
                            unpack(v)
                            if isinstance(v, Mapping) and "t" in v and "b64" in v
                            else v
                        )
                        for k, v in raw_map.items()
                    }
                elif (
                    isinstance(raw_resume, Mapping)
                    and "t" in raw_resume
                    and "b64" in raw_resume
                ):
                    res_val = unpack(raw_resume)
                else:
                    res_val = raw_resume
                cmd = (
                    res_val if isinstance(res_val, Command) else Command(resume=res_val)
                )
                ib = _merged_interrupt_before(graph, durable_pause_nodes)
                _invoke(
                    graph,
                    cmd,
                    config,
                    interrupt_after="*",
                    interrupt_before=ib,
                    max_concurrency=eff_mc,
                    context=context,
                    before_step=before_step,
                )
        elif kind == StepKind.TOOLS:
            node_name, tool_ns = request["node"], request.get("tool_ns", "")
            tn = _tool_nodes(graph).get(node_name)
            if tn is None:
                raise RunError(f"unknown tool node {node_name!r}")
            target_cfg = (
                _config(thread, None, tool_ns, run_config=run_config)
                if tool_ns
                else config
            )
            target_snap = _get_state(graph, target_cfg) if tool_ns else before
            pending_tasks = _pending_tool_tasks(graph, target_snap, node_name, tn)
            tool_calls = [call for _, calls, _ in pending_tasks for call in calls]
            task_input_types = [
                input_type for _, calls, input_type in pending_tasks for _ in calls
            ]
            durable_outputs = request["outputs"]
            if isinstance(durable_outputs, Mapping):
                dur_by_idx = {int(k): unpack(v) for k, v in durable_outputs.items()}
            elif len(durable_outputs) == len(tool_calls):
                dur_by_idx = {i: unpack(o) for i, o in enumerate(durable_outputs)}
            else:
                dur_by_idx = {}
                dur_iter = iter(durable_outputs)
                for idx, call in enumerate(tool_calls):
                    if tcfg.resolve(tn, node_name, call["name"], limits)[0]:
                        dur_by_idx[idx] = unpack(next(dur_iter))

            for idx, val in list(dur_by_idx.items()):
                if isinstance(val, dict) and DURABLE_TOOL_ERROR_KEY in val:
                    dur_by_idx[idx] = _format_failed_tool_message(
                        tn, tool_calls[idx], val[DURABLE_TOOL_ERROR_KEY]
                    )

            local_indices = [
                idx for idx in range(len(tool_calls)) if idx not in dur_by_idx
            ]
            local_by_idx: dict[int, Any] = {}
            if len(local_indices) <= 1 or eff_mc == 1:
                for idx in local_indices:
                    call = tool_calls[idx]
                    st, local_out = _exec_single_tool(
                        graph,
                        tn,
                        target_snap.values,
                        call,
                        thread,
                        tool_ns,
                        [],
                        handle_tool_errors=True,
                        run_config=run_config,
                        context=context,
                        node_name=node_name,
                        task_input_type=task_input_types[idx],
                    )
                    if st == StepStatus.INTERRUPTED:
                        raise RunError(
                            f"local tool {call.get('name')!r} called "
                            "interrupt(); mark it durable"
                        )
                    local_by_idx[idx] = local_out
            else:
                batch_sz = len(local_indices) if eff_mc is None else max(1, int(eff_mc))
                for start_i in range(0, len(local_indices), batch_sz):
                    chunk = local_indices[start_i : start_i + batch_sz]
                    with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
                        futs = [
                            (
                                idx,
                                pool.submit(
                                    _exec_single_tool,
                                    graph,
                                    tn,
                                    target_snap.values,
                                    tool_calls[idx],
                                    thread,
                                    tool_ns,
                                    [],
                                    handle_tool_errors=True,
                                    run_config=run_config,
                                    context=context,
                                    node_name=node_name,
                                    task_input_type=task_input_types[idx],
                                ),
                            )
                            for idx in chunk
                        ]
                        for idx, fut in futs:
                            st, local_out = fut.result()
                            if st == StepStatus.INTERRUPTED:
                                raise RunError(
                                    f"local tool {tool_calls[idx].get('name')!r} "
                                    "called interrupt(); mark it durable"
                                )
                            local_by_idx[idx] = local_out

            outputs = [
                dur_by_idx[idx] if idx in dur_by_idx else local_by_idx[idx]
                for idx in range(len(tool_calls))
            ]

            bare_node = node_name.rsplit(":", 1)[-1] if ":" in node_name else node_name
            task_graph = _tool_task_graph(graph, node_name)
            before_tasks = getattr(target_snap, "tasks", None) or ()
            uncompleted = [
                t for t in before_tasks if getattr(t, "result", None) is None
            ]
            offset = 0
            for task_id, calls, input_type in pending_tasks:
                task_outputs = outputs[offset : offset + len(calls)]
                offset += len(calls)
                combined = tn._combine_tool_outputs(task_outputs, input_type)
                _put_task_writes(task_graph, target_snap, bare_node, task_id, combined)
            if tool_ns:
                _invoke(
                    task_graph,
                    None,
                    target_cfg,
                    interrupt_after="*",
                    interrupt_before=[],
                    context=context,
                )
            elif len(uncompleted) == len(pending_tasks):
                _invoke(
                    graph,
                    None,
                    config,
                    interrupt_after="*",
                    interrupt_before=_merged_interrupt_before(
                        graph, durable_pause_nodes
                    ),
                    max_concurrency=eff_mc,
                    context=context,
                    before_step=before_step,
                )
        elif kind == StepKind.NODES:
            before_tasks = getattr(before, "tasks", None) or ()
            uncompleted = [
                t for t in before_tasks if getattr(t, "result", None) is None
            ]
            durable_outputs = request["outputs"]
            if isinstance(durable_outputs, Mapping):
                dur_by_idx = {int(k): unpack(v) for k, v in durable_outputs.items()}
            elif len(durable_outputs) == len(uncompleted):
                dur_by_idx = {i: unpack(o) for i, o in enumerate(durable_outputs)}
            else:
                dur_by_idx = {}
                dur_iter = iter(durable_outputs)
                for idx, t in enumerate(uncompleted):
                    if ncfg.resolve(graph, t.name, limits)[0]:
                        dur_by_idx[idx] = unpack(next(dur_iter))

            for idx, t in enumerate(uncompleted):
                if idx in dur_by_idx:
                    _put_task_writes(graph, before, t.name, t.id, dur_by_idx[idx])

            remaining_tnodes = [
                t
                for idx, t in enumerate(uncompleted)
                if idx not in dur_by_idx and t.name in durable_tnodes
            ]
            if not remaining_tnodes and not request.get("partial"):
                _invoke(
                    graph,
                    None,
                    config,
                    interrupt_after="*",
                    interrupt_before=[],
                    max_concurrency=eff_mc,
                    context=context,
                    before_step=before_step,
                )
        elif kind == StepKind.STEP:
            ib = _merged_interrupt_before(graph, durable_pause_nodes)
            _invoke(
                graph,
                None,
                config,
                interrupt_after="*",
                interrupt_before=ib,
                max_concurrency=eff_mc,
                context=context,
                before_step=before_step,
            )
        else:
            raise RunError(f"unknown step kind {kind!r}")

    if isinstance(saver, InMemorySaver):
        _remember(saver, "_durable_completed_expects", (thread, expect))
    after = _get_state(graph, config, subgraphs=has_subgraphs)
    return _report(
        graph,
        after,
        limits,
        tool_config=tcfg,
        node_config=ncfg,
        max_concurrency=eff_mc,
        node_max_concurrency=eff_nmc,
        version=req_version,
        root_graph=root_graph,
    )


def _extract_failure_info(exc: BaseException) -> dict[str, str]:
    """Extract `(error_type, message)` from a `TaskFailedError` or regular Python.

    exception.
    """
    details = getattr(exc, "details", None)
    if (
        details is not None
        and hasattr(details, "message")
        and hasattr(details, "error_type")
    ):
        return {
            "error_type": str(details.error_type or "Exception"),
            "message": str(details.message or ""),
        }
    message = str(exc)
    handled = getattr(exc, "_durable_tool_error_info", None)
    if handled is not None:
        message += _DURABLE_ERROR_MARKER + json.dumps(handled) + "]"
    return {"error_type": type(exc).__name__, "message": message}


def _extract_event_upgrade(event: Any) -> tuple[str | None, str | None, Any]:
    """Extract `(version, as_node, migrate_values)` from a resume external event.

    payload.
    """
    if not isinstance(event, Mapping):
        return None, None, None
    upg = event.get("upgrade")
    if isinstance(upg, Mapping):
        new_ver = upg.get("version") or event.get("version")
        as_node = upg.get("as_node") or event.get("as_node")
        mig_vals = (
            upg.get("migrate_values")
            if "migrate_values" in upg
            else (upg.get("values") if "values" in upg else event.get("migrate_values"))
        )
    elif isinstance(upg, str):
        new_ver = upg
        as_node = event.get("as_node")
        mig_vals = event.get("migrate_values")
    else:
        new_ver = event.get("version")
        as_node = event.get("as_node")
        mig_vals = event.get("migrate_values")
    return (str(new_ver) if new_ver is not None else None), as_node, mig_vals


def _build_can_payload(
    *,
    thread: str,
    generation: int,
    steps: int,
    request: dict[str, Any],
    step_name: str,
    tool_name: str,
    node_name: str,
    version: str | None = None,
    target_version: str | None = None,
    upgrade_on_can: bool | None = None,
    max_concurrency: int | None = None,
    node_max_concurrency: int | None = None,
    run_config: Any = None,
    context_blob: Any = None,
    checkpoint_bundle: Any = None,
) -> dict[str, Any]:
    """Construct a complete `continue_as_new` payload preserving all orchestration.

    state and options.
    """
    return dict(
        OrchestratorState(
            thread=thread,
            gen=generation - 1,
            steps=steps,
            request=request,
            step_name=step_name,
            tool_name=tool_name,
            node_name=node_name,
            version=version,
            target_version=target_version,
            upgrade_on_can=bool(upgrade_on_can),
            max_concurrency=max_concurrency,
            node_max_concurrency=node_max_concurrency,
            run_config=run_config,
            context_blob=context_blob,
            checkpoint_bundle=checkpoint_bundle,
        ).to_can_payload(target_version=target_version)
    )


def _is_valid_packed_blob(val: Any) -> bool:
    """Return whether `val` is a valid `PackedBlob` dict."""
    if not isinstance(val, Mapping):
        return False
    if "__resume_map__" in val and isinstance(val["__resume_map__"], Mapping):
        return all(_is_valid_packed_blob(v) for v in val["__resume_map__"].values())
    kind = val.get("t")
    b64 = val.get("b64")
    return (
        isinstance(kind, str)
        and kind in ("json", "msgpack", "null", "bytes", "empty")
        and isinstance(b64, str)
    )


def _cap_interrupt_for_status(intr: Mapping[str, Any], limit: int) -> dict[str, Any]:
    """Strip `value` and mark `oversized=True` when a tool/node interrupt exceeds `limit`."""  # noqa: E501
    out = dict(intr)
    if "value" in out and _size(out["value"]) > limit:
        out.pop("value", None)
        out["oversized"] = True
    return out


def _wait_for_resume_event(
    ctx: task.OrchestrationContext,
    intr_id: str,
    expected_ckpt: str | None,
    used: int,
) -> Generator[task.Task[Any], Any, tuple[dict[str, Any], int]]:
    """Wait on `'resume:<intr_id>'`, discarding malformed or stale duplicate retry events."""  # noqa: E501
    while True:
        event = yield ctx.wait_for_external_event(RESUME_EVENT_PREFIX + intr_id)
        used += _size(event)
        if not isinstance(event, Mapping):
            continue
        if "value" not in event or not _is_valid_packed_blob(event.get("value")):
            continue
        ev_ckpt = event.get("ckpt")
        if (
            ev_ckpt is not None
            and expected_ckpt is not None
            and ev_ckpt != expected_ckpt
        ):
            continue
        return dict(event), used


def make_orchestrator(
    limits: Limits,
    *,
    name: str | None = None,
) -> Callable[
    [task.OrchestrationContext, OrchestratorInput | dict[str, Any]],
    Generator[task.Task[Any], Any, dict[str, Any] | None],
]:
    """Build the Durable Task orchestrator generator function parameterized by.

    `limits`.
    """

    def run_graph_orchestrator(
        ctx: task.OrchestrationContext,
        inp: OrchestratorInput | dict[str, Any],
    ) -> Generator[task.Task[Any], Any, dict[str, Any] | None]:
        state = OrchestratorState.from_payload(inp)
        state.apply_step_options_to(state.request)

        def set_custom_status(status: dict[str, Any] | None) -> None:
            ctx.set_custom_status(
                status if status is not None else {"state": "running"}
            )

        def call_activity(
            name: str,
            *,
            input: dict[str, Any],
            retry_policy: task.RetryPolicy,
        ) -> task.Task[Any]:
            _check_payload(input, f"{name} input")
            return ctx.call_activity(name, input=input, retry_policy=retry_policy)

        eff_limits = state.effective_limits(limits)
        retry = task.RetryPolicy(
            first_retry_interval=timedelta(seconds=1),
            max_number_of_attempts=eff_limits.retries,
        )
        used, in_generation = 0, 0
        while True:
            if state.steps >= eff_limits.max_steps:
                raise RunError(f"step limit {eff_limits.max_steps} reached")
            report = yield call_activity(
                state.active_step_name, input=state.request, retry_policy=retry
            )
            if report.get("status") == StepStatus.ERROR:
                raise RunError(report["error"])
            if "checkpoint_bundle" in report:
                state.checkpoint_bundle = report["checkpoint_bundle"]
            if report.get("target_version"):
                state.target_version = str(report["target_version"])
            state.steps += 1
            in_generation += 1
            used += _size(state.request) + _size(report)

            while report["status"] in (StepStatus.TOOLS, StepStatus.NODES):
                if state.steps >= eff_limits.max_steps:
                    raise RunError(f"step limit {eff_limits.max_steps} reached")
                jumped_subtask = False
                if report["status"] == StepStatus.TOOLS:
                    all_calls = report["tool_calls"]
                    durable_calls = [tc for tc in all_calls if tc.get("durable", True)]
                    eff_mc = (
                        state.max_concurrency
                        if state.max_concurrency is not None
                        else report.get("max_concurrency")
                    )
                    if eff_mc is not None:
                        eff_mc = max(1, int(eff_mc))
                    resumes_by_idx: dict[int, list[Any]] = {
                        tc["index"]: [] for tc in durable_calls
                    }
                    outputs_by_idx: dict[int, Any] = {}
                    batch_size = len(durable_calls) if eff_mc is None else eff_mc
                    batches = (
                        [
                            durable_calls[i : i + batch_size]
                            for i in range(0, len(durable_calls), batch_size)
                        ]
                        if batch_size > 0
                        else []
                    )
                    for batch in batches:
                        pending_calls = list(batch)
                        while pending_calls:
                            batch_tasks: list[task.Task[Any]] = []
                            for tc in pending_calls:
                                attempts = tc.get(
                                    "retries",
                                    eff_limits.tool_retries or eff_limits.retries,
                                )
                                t_retry = task.RetryPolicy(
                                    first_retry_interval=timedelta(seconds=1),
                                    max_number_of_attempts=max(1, int(attempts)),
                                )
                                t_req: dict[str, Any] = {
                                    "thread": state.thread,
                                    "ckpt": report["ckpt"],
                                    "node": report["node"],
                                    "call_id": tc["call_id"],
                                    "index": tc["index"],
                                    "call": tc["call"],
                                }
                                for key in ("task_id", "input_type"):
                                    if key in tc:
                                        t_req[key] = tc[key]
                                if report.get("tool_ns"):
                                    t_req["tool_ns"] = report["tool_ns"]
                                if report.get("tool_ckpt"):
                                    t_req["tool_ckpt"] = report["tool_ckpt"]
                                state.apply_context_to(t_req)
                                resumes = resumes_by_idx[tc["index"]]
                                if resumes:
                                    t_req["resumes"] = list(resumes)
                                batch_tasks.append(
                                    call_activity(
                                        state.active_tool_name,
                                        input=t_req,
                                        retry_policy=t_retry,
                                    )
                                )
                                in_generation += 1
                                used += _size(t_req)
                            batch_results = []
                            for tc_item, bt in zip(pending_calls, batch_tasks):
                                try:
                                    t_res = yield bt
                                except Exception as bt_exc:
                                    if not tc_item.get("handle_errors"):
                                        raise
                                    t_res = {
                                        "call_id": tc_item["call_id"],
                                        "status": StepStatus.OK,
                                        "output": pack(
                                            {
                                                DURABLE_TOOL_ERROR_KEY: (
                                                    _extract_failure_info(bt_exc)
                                                )
                                            }
                                        ),
                                    }
                                if t_res.get("status") == StepStatus.ERROR:
                                    raise RunError(t_res["error"])
                                batch_results.append(t_res)
                            interrupted_pairs: list[
                                tuple[dict[str, Any], dict[str, Any]]
                            ] = []
                            for tc, t_res in zip(pending_calls, batch_results):
                                used += _size(t_res)
                                if t_res.get("status") == StepStatus.INTERRUPTED:
                                    interrupted_pairs.append(
                                        (
                                            tc,
                                            _cap_interrupt_for_status(
                                                t_res["interrupt"],
                                                eff_limits.custom_status_limit,
                                            ),
                                        )
                                    )
                                else:
                                    outputs_by_idx[tc["index"]] = t_res["output"]
                            if interrupted_pairs:
                                last_t_event: dict[str, Any] = {}
                                if len(interrupted_pairs) == 1:
                                    tc, pending = interrupted_pairs[0]
                                    r_cnt = len(resumes_by_idx[tc["index"]])
                                    cur_t_ckpt = f"{report['ckpt']}:t:{r_cnt}"
                                    set_custom_status(
                                        {
                                            "pending": pending,
                                            "interrupts": [pending],
                                            "step": state.steps + r_cnt,
                                            "ckpt": cur_t_ckpt,
                                        }
                                    )
                                    event, used = yield from _wait_for_resume_event(
                                        ctx, pending["id"], cur_t_ckpt, used
                                    )
                                    set_custom_status(None)
                                    last_t_event = event
                                    resumes_by_idx[tc["index"]].append(event["value"])
                                else:
                                    resumed_ids: set[str] = set()
                                    for (
                                        tc,
                                        intr,
                                    ) in interrupted_pairs:
                                        remaining = [
                                            i
                                            for _, i in interrupted_pairs
                                            if i["id"] not in resumed_ids
                                        ]
                                        r_cnt = len(resumes_by_idx[tc["index"]])
                                        cur_t_ckpt = f"{report['ckpt']}:t:{r_cnt}"  # noqa: E501
                                        set_custom_status(
                                            {
                                                "pending": intr,
                                                "interrupts": remaining,
                                                "step": state.steps + r_cnt,
                                                "ckpt": cur_t_ckpt,
                                            }
                                        )
                                        (
                                            event,
                                            used,
                                        ) = yield from _wait_for_resume_event(  # noqa: E501
                                            ctx, intr["id"], cur_t_ckpt, used
                                        )
                                        last_t_event = event
                                        resumed_ids.add(intr["id"])
                                        resumes_by_idx[tc["index"]].append(
                                            event["value"]
                                        )
                                    set_custom_status(None)
                                new_ver, upg_as_node, upg_vals = _extract_event_upgrade(
                                    last_t_event
                                )
                                if (
                                    (new_ver is not None and new_ver != state.version)
                                    or upg_as_node is not None
                                    or upg_vals is not None
                                ):
                                    mig_req: dict[str, Any] = {
                                        "thread": state.thread,
                                        "kind": StepKind.MIGRATE,
                                        "expect": report["ckpt"],
                                        "migrate_from": state.version,
                                        "version": new_ver or state.version,
                                        "resume": last_t_event.get("value"),
                                    }
                                    if upg_as_node is not None:
                                        mig_req["as_node"] = upg_as_node
                                    if upg_vals is not None:
                                        mig_req["migrate_values"] = upg_vals
                                    state.version = new_ver or state.version
                                    state.apply_context_to(mig_req)
                                    mig_rep = yield call_activity(
                                        state.active_step_name,
                                        input=mig_req,
                                        retry_policy=retry,
                                    )
                                    if mig_rep.get("status") == StepStatus.ERROR:
                                        raise RunError(mig_rep["error"])
                                    if "checkpoint_bundle" in mig_rep:
                                        state.checkpoint_bundle = mig_rep[
                                            "checkpoint_bundle"
                                        ]
                                    if mig_rep.get("target_version"):
                                        state.target_version = str(
                                            mig_rep["target_version"]
                                        )
                                    state.steps += 1
                                    in_generation += 1
                                    used += _size(mig_req) + _size(mig_rep)
                                    if mig_rep.get("jumped"):
                                        report = mig_rep
                                        jumped_subtask = True
                                        break
                                    report = {**report, "ckpt": mig_rep["ckpt"]}
                                pending_calls = [tc for tc, _ in interrupted_pairs]
                            else:
                                pending_calls = []
                        if jumped_subtask:
                            break

                    if jumped_subtask:
                        if (
                            used >= eff_limits.byte_budget
                            or in_generation >= eff_limits.steps_per_generation
                        ):
                            break
                        continue
                    tool_outputs = [outputs_by_idx[tc["index"]] for tc in durable_calls]
                    tool_outputs_by_idx = {str(k): v for k, v in outputs_by_idx.items()}
                    commit_req: dict[str, Any] = {
                        "thread": state.thread,
                        "kind": StepKind.TOOLS,
                        "expect": report["ckpt"],
                        "node": report["node"],
                        "outputs": tool_outputs
                        if len(durable_calls) == len(all_calls)
                        else tool_outputs_by_idx,
                    }
                    if report.get("tool_ns"):
                        commit_req["tool_ns"] = report["tool_ns"]
                    if eff_mc is not None:
                        commit_req["max_concurrency"] = eff_mc
                else:
                    all_tasks = report["node_tasks"]
                    durable_tasks = [nt for nt in all_tasks if nt.get("durable", True)]
                    eff_nmc = (
                        state.node_max_concurrency
                        if state.node_max_concurrency is not None
                        else (
                            state.max_concurrency
                            if state.max_concurrency is not None
                            else report.get("max_concurrency")
                        )
                    )
                    if eff_nmc is not None:
                        eff_nmc = max(1, int(eff_nmc))
                    n_resumes_by_idx: dict[int, list[Any]] = {
                        nt["index"]: [] for nt in durable_tasks
                    }
                    n_outputs_by_idx: dict[int, Any] = {}
                    n_batch_size = len(durable_tasks) if eff_nmc is None else eff_nmc
                    n_batches = (
                        [
                            durable_tasks[i : i + n_batch_size]
                            for i in range(0, len(durable_tasks), n_batch_size)
                        ]
                        if n_batch_size > 0
                        else []
                    )
                    for batch in n_batches:
                        pending_nodes = list(batch)
                        while pending_nodes:
                            batch_tasks = []
                            for nt in pending_nodes:
                                attempts = nt.get(
                                    "retries",
                                    eff_limits.node_retries or eff_limits.retries,
                                )
                                n_retry = task.RetryPolicy(
                                    first_retry_interval=timedelta(seconds=1),
                                    max_number_of_attempts=max(1, int(attempts)),
                                )
                                n_req: dict[str, Any] = {
                                    "thread": state.thread,
                                    "ckpt": report["ckpt"],
                                    "call_id": nt["call_id"],
                                    "index": nt["index"],
                                    "task_id": nt["task_id"],
                                    "name": nt["name"],
                                }
                                if "input" in nt:
                                    n_req["input"] = nt["input"]
                                state.apply_context_to(n_req)
                                resumes = n_resumes_by_idx[nt["index"]]
                                if resumes:
                                    n_req["resumes"] = list(resumes)
                                batch_tasks.append(
                                    call_activity(
                                        state.active_node_name,
                                        input=n_req,
                                        retry_policy=n_retry,
                                    )
                                )
                                in_generation += 1
                                used += _size(n_req)
                            interrupted_nodes: list[
                                tuple[dict[str, Any], dict[str, Any]]
                            ] = []
                            try:
                                for nt, bt in zip(pending_nodes, batch_tasks):
                                    n_res = yield bt
                                    if n_res.get("status") == StepStatus.ERROR:
                                        raise RunError(n_res["error"])
                                    used += _size(n_res)
                                    if n_res.get("status") == StepStatus.INTERRUPTED:
                                        interrupted_nodes.append(
                                            (
                                                nt,
                                                _cap_interrupt_for_status(
                                                    n_res["interrupt"],
                                                    eff_limits.custom_status_limit,
                                                ),
                                            )
                                        )
                                    else:
                                        n_outputs_by_idx[nt["index"]] = n_res["output"]
                            except Exception:
                                if n_outputs_by_idx:
                                    partial_req: dict[str, Any] = {
                                        "thread": state.thread,
                                        "kind": StepKind.NODES,
                                        "expect": report["ckpt"],
                                        "outputs": {
                                            str(k): v
                                            for k, v in n_outputs_by_idx.items()
                                        },
                                        "partial": True,
                                    }
                                    state.apply_context_to(partial_req)
                                    yield call_activity(
                                        state.active_step_name,
                                        input=partial_req,
                                        retry_policy=retry,
                                    )
                                raise
                            if interrupted_nodes:
                                last_n_event: dict[str, Any] = {}
                                if len(interrupted_nodes) == 1:
                                    nt, pending = interrupted_nodes[0]
                                    r_cnt = len(n_resumes_by_idx[nt["index"]])
                                    cur_n_ckpt = f"{report['ckpt']}:n:{r_cnt}"
                                    set_custom_status(
                                        {
                                            "pending": pending,
                                            "interrupts": [pending],
                                            "step": state.steps + r_cnt,
                                            "ckpt": cur_n_ckpt,
                                        }
                                    )
                                    event, used = yield from _wait_for_resume_event(
                                        ctx, pending["id"], cur_n_ckpt, used
                                    )
                                    set_custom_status(None)
                                    last_n_event = event
                                    n_resumes_by_idx[nt["index"]].append(event["value"])
                                else:
                                    resumed_ids = set()
                                    for (
                                        nt,
                                        intr,
                                    ) in interrupted_nodes:
                                        remaining = [
                                            i
                                            for _, i in interrupted_nodes
                                            if i["id"] not in resumed_ids
                                        ]
                                        r_cnt = len(n_resumes_by_idx[nt["index"]])
                                        cur_n_ckpt = f"{report['ckpt']}:n:{r_cnt}"  # noqa: E501
                                        set_custom_status(
                                            {
                                                "pending": intr,
                                                "interrupts": remaining,
                                                "step": state.steps + r_cnt,
                                                "ckpt": cur_n_ckpt,
                                            }
                                        )
                                        (
                                            event,
                                            used,
                                        ) = yield from _wait_for_resume_event(  # noqa: E501
                                            ctx, intr["id"], cur_n_ckpt, used
                                        )
                                        last_n_event = event
                                        resumed_ids.add(intr["id"])
                                        n_resumes_by_idx[nt["index"]].append(
                                            event["value"]
                                        )
                                    set_custom_status(None)
                                new_ver, upg_as_node, upg_vals = _extract_event_upgrade(
                                    last_n_event
                                )
                                if (
                                    (new_ver is not None and new_ver != state.version)
                                    or upg_as_node is not None
                                    or upg_vals is not None
                                ):
                                    mig_req = {
                                        "thread": state.thread,
                                        "kind": StepKind.MIGRATE,
                                        "expect": report["ckpt"],
                                        "migrate_from": state.version,
                                        "version": new_ver or state.version,
                                        "resume": last_n_event.get("value"),
                                    }
                                    if upg_as_node is not None:
                                        mig_req["as_node"] = upg_as_node
                                    if upg_vals is not None:
                                        mig_req["migrate_values"] = upg_vals
                                    state.version = new_ver or state.version
                                    state.apply_context_to(mig_req)
                                    mig_rep = yield call_activity(
                                        state.active_step_name,
                                        input=mig_req,
                                        retry_policy=retry,
                                    )
                                    if mig_rep.get("status") == StepStatus.ERROR:
                                        raise RunError(mig_rep["error"])
                                    if "checkpoint_bundle" in mig_rep:
                                        state.checkpoint_bundle = mig_rep[
                                            "checkpoint_bundle"
                                        ]
                                    if mig_rep.get("target_version"):
                                        state.target_version = str(
                                            mig_rep["target_version"]
                                        )
                                    state.steps += 1
                                    in_generation += 1
                                    used += _size(mig_req) + _size(mig_rep)
                                    if mig_rep.get("jumped"):
                                        report = mig_rep
                                        jumped_subtask = True
                                        break
                                    report = {
                                        **report,
                                        "ckpt": mig_rep["ckpt"],
                                    }
                                pending_nodes = [nt for nt, _ in interrupted_nodes]
                            else:
                                pending_nodes = []
                        if jumped_subtask:
                            break

                    if jumped_subtask:
                        if (
                            used >= eff_limits.byte_budget
                            or in_generation >= eff_limits.steps_per_generation
                        ):
                            break
                        continue
                    node_outputs = [
                        n_outputs_by_idx[nt["index"]] for nt in durable_tasks
                    ]
                    node_outputs_by_idx = {
                        str(k): v for k, v in n_outputs_by_idx.items()
                    }
                    commit_req = {
                        "thread": state.thread,
                        "kind": StepKind.NODES,
                        "expect": report["ckpt"],
                        "outputs": node_outputs
                        if len(durable_tasks) == len(all_tasks)
                        else node_outputs_by_idx,
                    }
                    if eff_nmc is not None:
                        commit_req["node_max_concurrency"] = eff_nmc
                    if state.max_concurrency is not None:
                        commit_req["max_concurrency"] = state.max_concurrency

                state.apply_context_to(commit_req)
                report = yield call_activity(
                    state.active_step_name, input=commit_req, retry_policy=retry
                )
                if report.get("status") == StepStatus.ERROR:
                    raise RunError(report["error"])
                if "checkpoint_bundle" in report:
                    state.checkpoint_bundle = report["checkpoint_bundle"]
                if report.get("target_version"):
                    state.target_version = str(report["target_version"])
                state.steps += 1
                in_generation += 1
                used += _size(commit_req) + _size(report)
                if (
                    used >= eff_limits.byte_budget
                    or in_generation >= eff_limits.steps_per_generation
                ):
                    break

            if report["status"] == StepStatus.DONE:
                out: dict[str, Any] = {
                    "status": StepStatus.DONE,
                    "thread": state.thread,
                    "steps": state.steps,
                    "gen": state.gen,
                    "ckpt": report["ckpt"],
                }
                if state.version is not None:
                    out["version"] = state.version
                if "result" in report:
                    out["result"] = report["result"]
                if report.get("output") is not None and not state.checkpoint_bundle:
                    out["output"] = report["output"]
                if state.checkpoint_bundle is not None:
                    out["checkpoint_bundle"] = state.checkpoint_bundle
                _check_payload(out, "orchestrator output")
                set_custom_status({"state": "completed"})
                return out
            if report["status"] == StepStatus.INTERRUPTED:
                interrupts = report.get("interrupts") or [report["interrupt"]]
                last_event: dict[str, Any] = {}
                cur_ckpt = report.get("ckpt")
                if len(interrupts) == 1:
                    pending = interrupts[0]
                    set_custom_status(
                        {
                            "pending": pending,
                            "interrupts": interrupts,
                            "step": state.steps,
                            "ckpt": cur_ckpt,
                        }
                    )
                    event, used = yield from _wait_for_resume_event(
                        ctx, pending["id"], cur_ckpt, used
                    )
                    set_custom_status(None)
                    last_event = event
                    resume_val = event["value"]
                else:
                    resume_map: dict[str, Any] = {}
                    for intr in interrupts:
                        remaining = [i for i in interrupts if i["id"] not in resume_map]
                        set_custom_status(
                            {
                                "pending": intr,
                                "interrupts": remaining,
                                "step": state.steps,
                                "ckpt": cur_ckpt,
                            }
                        )
                        event, used = yield from _wait_for_resume_event(
                            ctx, intr["id"], cur_ckpt, used
                        )
                        last_event = event
                        resume_map[intr["id"]] = event["value"]
                    set_custom_status(None)
                    resume_val = {"__resume_map__": resume_map}
                state.request = {
                    "thread": state.thread,
                    "kind": StepKind.RESUME,
                    "expect": report["ckpt"],
                    "resume": resume_val,
                }
                new_ver, upg_as_node, upg_vals = _extract_event_upgrade(last_event)
                if new_ver is not None and new_ver != state.version:
                    state.request["migrate_from"] = state.version
                    state.version = new_ver
                if upg_as_node is not None:
                    state.request["as_node"] = upg_as_node
                if upg_vals is not None:
                    state.request["migrate_values"] = upg_vals
            elif report["status"] in (
                StepStatus.TOOLS,
                StepStatus.NODES,
            ):
                state.request = {
                    "thread": state.thread,
                    "kind": StepKind.INSPECT,
                    "expect": report["ckpt"],
                }
            else:
                state.request = {
                    "thread": state.thread,
                    "kind": StepKind.STEP,
                    "expect": report["ckpt"],
                }
            state.apply_step_options_to(state.request)
            if (
                used >= eff_limits.byte_budget
                or in_generation >= eff_limits.steps_per_generation
            ):
                eff_target_ver = report.get("target_version") or (
                    state.target_version if state.upgrade_on_can else None
                )
                if (
                    eff_target_ver is not None
                    and eff_target_ver != state.version
                    and "migrate_from" not in state.request
                ):
                    state.request["migrate_from"] = state.version
                    state.version = eff_target_ver
                    state.request["version"] = state.version
                ctx.continue_as_new(
                    state.to_can_payload(target_version=eff_target_ver),
                    save_events=False,
                )
                return None

    if name is not None:
        run_graph_orchestrator.__name__ = name
    return run_graph_orchestrator
