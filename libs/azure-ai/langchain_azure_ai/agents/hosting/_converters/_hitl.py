# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Human-in-the-loop translation between LangGraph and the Responses API.

LangGraph pauses execution when a node calls ``langgraph.types.interrupt``.
The pause is checkpointed and surfaced on
:attr:`langgraph.types.StateSnapshot.interrupts`. Resume happens by
invoking the graph again with a :class:`langgraph.types.Command` carrying
``resume`` / ``update`` / ``goto`` fields.

We map this onto the OpenAI Responses API by emitting *two* output items
per pending interrupt so off-the-shelf clients can drive resume through
either of two standard channels:

1. A ``function_call`` output item named
   :data:`HITL_FUNCTION_NAME` with ``call_id == interrupt.id``. The
   ``arguments`` field carries the ``{"interrupt_id", "value"}`` envelope
   (JSON-encoded).
2. An ``mcp_approval_request`` output item with a storage-compatible generated
    ``mcpr_*`` id, ``server_label == "langgraph"``, the same ``name``, and the
    same ``arguments`` envelope.

Both items carry the same ``interrupt.id`` in their arguments. The client
resumes by posting either:

* a ``function_call_output`` input item (rich payload — can carry
  ``{"resume"|"update"|"goto"}``), or
* an ``mcp_approval_response`` input item (approve-only — ``approve=true``
  resumes with the original interrupt value echoed back; ``approve=false``
  is surfaced to the host as a rejection signal).

When both shapes target the same ``interrupt.id`` in one request,
``function_call_output`` wins (it carries the richer payload) and a
warning is logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Final, TypeGuard

from azure.ai.agentserver.responses import ResponseEventStream
from azure.ai.agentserver.responses.models import (
    FunctionCallOutputItemParam,
    ItemFunctionToolCall,
    MCPApprovalResponse,
)
from azure.ai.agentserver.responses.models._generated import (
    OutputItemMcpApprovalRequest,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.types import Command, Interrupt

if TYPE_CHECKING:
    from langgraph.types import StateSnapshot

logger = logging.getLogger(__name__)

HITL_FUNCTION_NAME: Final[str] = "__hosted_agent_adapter_interrupt__"
"""Reserved ``function_call.name`` / ``mcp_approval_request.name`` used to
surface a LangGraph interrupt.

The string value matches the ``HUMAN_IN_THE_LOOP_FUNCTION_NAME`` used by
``azure-ai-agentserver-langgraph`` so clients can share the same
discriminator across both hosts. Treat the literal as opaque and match
on it via this symbol.
"""


def _is_function_call(item: Any) -> TypeGuard[ItemFunctionToolCall]:
    return isinstance(item, dict) and item.get("type") == "function_call"


def _is_function_call_output(item: Any) -> TypeGuard[FunctionCallOutputItemParam]:
    return (
        isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and isinstance(item.get("call_id"), str)
        and "output" in item
    )


def _is_mcp_approval_response(item: Any) -> TypeGuard[MCPApprovalResponse]:
    return (
        isinstance(item, dict)
        and item.get("type") == "mcp_approval_response"
        and isinstance(item.get("approval_request_id"), str)
        and isinstance(item.get("approve"), bool)
    )


HITL_MCP_SERVER_LABEL: Final[str] = "langgraph"
"""``server_label`` stamped on the ``mcp_approval_request`` we emit.

We borrow the MCP approval item type as a generic approval channel
(mirroring Microsoft Agent Framework's `foundry_hosting`). The label
exists so clients can discriminate our HITL items from real MCP
approval requests at a glance.
"""


async def detect_pending_interrupts(
    graph: Runnable[Any, Any], config: RunnableConfig
) -> tuple[Interrupt, ...]:
    """Return the interrupts pending on the checkpointed state, if any.

    ``StateSnapshot.interrupts`` accumulates every interrupt recorded on the
    checkpoint and is *not* pruned as they are answered, so after a partial
    resume of parallel interrupts it may still report the ones already
    satisfied. Deriving the list from ``StateSnapshot.tasks`` and skipping
    tasks that produced node output keeps the initial lookup aligned with
    what is outstanding in the common case. After a graph invocation,
    :func:`track_pending_interrupts` provides the authoritative active set.

    Args:
        graph: The runnable to inspect. Runnables without an ``aget_state``
            method have no checkpointed interrupts and return an empty tuple.
        config: The :class:`RunnableConfig` identifying the thread.

    Returns:
        A tuple of :class:`Interrupt` objects (empty when none pending or
        when the graph has no checkpointer attached).
    """
    get_state = getattr(graph, "aget_state", None)
    if get_state is None:
        return ()
    try:
        snapshot: "StateSnapshot | None" = await get_state(config)
    except Exception:  # noqa: BLE001
        # No checkpointer / unknown thread / provider error — treat as
        # "nothing pending" and let the regular path run.
        logger.debug("aget_state failed; assuming no pending interrupts", exc_info=True)
        return ()
    if snapshot is None:
        return ()
    tasks = tuple(getattr(snapshot, "tasks", None) or ())
    pending_tasks = tuple(
        task for task in tasks if getattr(task, "result", None) is None
    )
    if not pending_tasks:
        # A node with several sequential interrupt() calls can retain a partial
        # empty result while pausing again. In that case there is no result-less
        # sibling, so its current interrupt is still active.
        pending_tasks = tasks

    seen: set[str] = set()
    pending: list[Interrupt] = []
    for task in pending_tasks:
        for it in getattr(task, "interrupts", None) or ():
            if isinstance(it, Interrupt) and it.id not in seen:
                seen.add(it.id)
                pending.append(it)
    return tuple(pending)


async def track_pending_interrupts(
    graph_stream: AsyncIterator[Any], pending: list[Interrupt]
) -> AsyncIterator[Any]:
    """Pass through a graph stream while recording its active interrupts.

    LangGraph's ``updates.__interrupt__`` payload is the authoritative active
    set for the current invocation. Unlike checkpoint task history, it does
    not retain a parallel sibling after that sibling has been answered.

    Args:
        graph_stream: A multi-mode LangGraph stream containing ``updates``.
        pending: Mutable output list containing the interrupts observed during
            this invocation, deduplicated by id.

    Yields:
        Every input chunk unchanged and in its original order.
    """
    active_by_id: dict[str, Interrupt] = {}
    pending.clear()
    async for chunk in graph_stream:
        if (
            isinstance(chunk, tuple)
            and len(chunk) == 2
            and chunk[0] == "updates"
            and isinstance(chunk[1], dict)
            and "__interrupt__" in chunk[1]
        ):
            raw_interrupts = chunk[1]["__interrupt__"]
            if isinstance(raw_interrupts, Interrupt):
                raw_interrupts = (raw_interrupts,)
            if not isinstance(raw_interrupts, (list, tuple)):
                raw_interrupts = ()
            active_by_id.update(
                (interrupt.id, interrupt)
                for interrupt in raw_interrupts
                if isinstance(interrupt, Interrupt)
            )
            pending[:] = active_by_id.values()
        yield chunk


def interrupt_arguments_json(interrupt: Interrupt) -> str:
    """Render the ``{"interrupt_id", "value"}`` envelope as a JSON string.

    The envelope is used as the ``arguments`` payload on both the
    ``function_call`` and ``mcp_approval_request`` items emitted by
    :func:`emit_interrupts`. Wrapping the raw value lets clients render
    HITL prompts uniformly across the two channels and lets the
    approval-response decode path validate the request id without
    server-side storage.

    Non-serializable interrupt values fall back to their ``str()``
    representation so emission cannot fail at the wire layer.

    Args:
        interrupt: The interrupt to encode.

    Returns:
        The JSON string to use as the wire ``arguments`` field.
    """
    try:
        return json.dumps({"interrupt_id": interrupt.id, "value": interrupt.value})
    except (TypeError, ValueError):
        logger.warning("Interrupt value not JSON-serializable; falling back to str().")
        return json.dumps({"interrupt_id": interrupt.id, "value": str(interrupt.value)})


def interrupt_output_items(
    interrupts: Iterable[Interrupt],
) -> list[dict[str, Any]]:
    """Build portable Responses-style output items for pending interrupts.

    Unlike :func:`emit_interrupts`, this helper does not require a Responses
    event stream, so generic protocol adapters can expose the same HITL wire
    shapes. Item IDs are deterministic to remain stable across polling and
    process recovery.
    """
    output: list[dict[str, Any]] = []
    for interrupt in interrupts:
        if not isinstance(interrupt, Interrupt):
            continue
        suffix = _approval_id_suffix(interrupt.id)
        arguments = interrupt_arguments_json(interrupt)
        output.extend(
            [
                {
                    "type": "function_call",
                    "id": f"fc_{suffix}",
                    "call_id": interrupt.id,
                    "name": HITL_FUNCTION_NAME,
                    "arguments": arguments,
                    "status": "completed",
                },
                {
                    "type": "mcp_approval_request",
                    "id": f"mcpr_{suffix}",
                    "server_label": HITL_MCP_SERVER_LABEL,
                    "name": HITL_FUNCTION_NAME,
                    "arguments": arguments,
                },
            ]
        )
    return output


def hitl_call_ids(items: Sequence[Any]) -> frozenset[str]:
    """Return the ``call_id``s reserved by the HITL wire protocol.

    An id is reserved when *items* carries a ``function_call`` named
    :data:`HITL_FUNCTION_NAME` — the sentinel this host emits for a pending
    interrupt. That sentinel and the ``function_call_output`` answering it
    are transport plumbing rather than conversation content, so
    :func:`~._request.items_to_messages` drops both instead of replaying
    them to the model as a tool-call round trip.

    This matters beyond the resume turn itself. On the turn that consumes
    an interrupt the host already filters the pair through the resume
    path's consumed-id set, but a client that echoes prior output items
    back — the ordinary stateless Responses pattern — keeps re-sending the
    sentinel on every later turn, when nothing is pending and no
    consumed-id set exists.

    Only the ``function_call`` side carries the discriminating name. A
    ``function_call_output`` arriving without it cannot be recognised
    here, but an unpaired one is already dropped as an orphan
    ``ToolMessage``.

    Args:
        items: Resolved input items from the request and/or history.

    Returns:
        The reserved ``call_id``s, empty when the stream carries none.
    """
    reserved: set[str] = set()
    for item in items:
        if not _is_function_call(item):
            continue
        if item.get("name") != HITL_FUNCTION_NAME:
            continue
        call_id = item.get("call_id")
        if isinstance(call_id, str) and call_id:
            reserved.add(call_id)
    return frozenset(reserved)


def parse_resume_command(
    items: Sequence[Any],
    pending: Sequence[Interrupt],
) -> tuple[Command | None, frozenset[str]]:
    """Build a resume :class:`Command` from request input items, if present.

    Two input shapes are accepted, both keyed by ``interrupt.id``:

    * :class:`FunctionCallOutputItemParam` matched by ``call_id``. Its
      ``output`` field is decoded — a JSON object with any of
      ``{"resume", "update", "goto"}`` populates the :class:`Command`;
      anything else (string, malformed JSON, list of content parts) is
      treated as the raw resume value.
    * :class:`MCPApprovalResponse` matched by ``approval_request_id``.
      ``approve=True`` resumes with the original interrupt value;
      ``approve=False`` is *not* handled here — use
      :func:`detect_approval_rejection` to surface the rejection to the
      host.

    Conflict resolution: when both shapes target the same interrupt id
    in one request, the ``function_call_output`` wins (richer payload)
    and a warning is logged. This is a deliberate, deterministic
    departure from Agent Framework's order-dependent last-write-wins.

    Resume shape: with a single pending interrupt the resume value is
    passed through as-is (``Command(resume=value)``). With *parallel*
    pending interrupts LangGraph requires an id-keyed resume map — a
    bare value raises ``RuntimeError: When there are multiple pending
    interrupts, you must specify the interrupt id when resuming.`` — so
    every matched item that explicitly carries a resume value is folded
    into ``Command(resume={id: value})``. Answering only some of the pending
    interrupts is allowed; the unanswered ones stay paused.

    Args:
        items: Resolved input items from the request.
        pending: Pending interrupts on the graph's checkpointed state.

    Returns:
        A ``(command, consumed_call_ids)`` pair. ``command`` is ``None``
        when no matching resume item was found.
    """
    if not pending:
        return None, frozenset()

    pending_by_id: dict[str, Interrupt] = {it.id: it for it in pending}
    # interrupt id -> (decoded command, explicitly carries resume), in
    # first-seen order. ``Command.resume is None`` alone cannot distinguish
    # an omitted resume from an explicit ``{"resume": null}``.
    commands: dict[str, tuple[Command, bool]] = {}
    # interrupt id -> the wire id that produced it (the encoded ``mcpr_*``
    # id for approvals, the raw interrupt id for function call outputs).
    consumed: dict[str, str] = {}

    # Pass 1 — prefer function_call_output (richer payload).
    for item in items:
        if not _is_function_call_output(item):
            continue
        call_id = item["call_id"]
        if call_id not in pending_by_id or call_id in commands:
            continue
        decoded = _decode_command(item["output"])
        if decoded is None:
            continue
        command, has_resume = decoded
        _warn_if_competing_approval(items, call_id)
        commands[call_id] = (command, has_resume)
        consumed[call_id] = call_id

    # Pass 2 — fall back to mcp_approval_response (approve-only).
    for item in items:
        if not _is_mcp_approval_response(item):
            continue
        approval_id = item["approval_request_id"]
        interrupt_id = _interrupt_id_from_approval_id(approval_id, pending_by_id)
        interrupt_obj = pending_by_id.get(interrupt_id)
        if interrupt_obj is None or interrupt_id in commands:
            continue
        if not item["approve"]:
            # Rejection is surfaced via ``detect_approval_rejection``
            # rather than as a ``Command``. Skip it here.
            continue
        commands[interrupt_id] = (Command(resume=interrupt_obj.value), True)
        consumed[interrupt_id] = approval_id

    if not commands:
        return None, frozenset()

    consumed_ids = frozenset(consumed.values())
    if len(pending_by_id) == 1:
        # Exactly one pause — LangGraph accepts the bare resume value.
        interrupt_id, (command, has_resume) = next(iter(commands.items()))
        if has_resume and command.resume is None:
            # ``Command(resume=None)`` means no resume; an id-keyed map is
            # required to pass an explicit null value through to interrupt().
            command = Command(
                resume={interrupt_id: None},
                update=command.update,
                goto=command.goto,
            )
        return command, consumed_ids

    # Parallel pauses — LangGraph matches resume values by interrupt id.
    resume = {
        interrupt_id: command.resume
        for interrupt_id, (command, has_resume) in commands.items()
        if has_resume
    }
    decoded_commands = [command for command, _ in commands.values()]
    return (
        Command(
            resume=resume or None,
            update=_merge_command_updates(decoded_commands),
            goto=_merge_command_gotos(decoded_commands),
        ),
        consumed_ids,
    )


def detect_approval_rejection(
    items: Sequence[Any],
    pending: Sequence[Interrupt],
) -> str | None:
    """Return a human-readable message if the client rejected an interrupt.

    Scans for :class:`MCPApprovalResponse` items whose
    ``approval_request_id`` matches a pending interrupt and whose
    ``approve`` is ``False``. The first match wins; subsequent rejections
    are ignored.

    The host's :meth:`handle_create` calls this *before* attempting to
    resume so a rejection short-circuits the turn into
    ``response.failed`` instead of being silently dropped.

    Args:
        items: Resolved input items from the request.
        pending: Pending interrupts on the graph's checkpointed state.

    Returns:
        The rejection message (including the rejected interrupt id and
        any client-supplied ``reason``), or ``None`` when no rejection
        was found.
    """
    if not pending:
        return None
    pending_ids = {it.id for it in pending}
    function_output_ids = {
        item["call_id"]
        for item in items
        if _is_function_call_output(item)
        and item["call_id"] in pending_ids
        and _decode_command(item["output"]) is not None
    }
    for item in items:
        if not _is_mcp_approval_response(item):
            continue
        if item["approve"]:
            continue
        approval_id = item["approval_request_id"]
        interrupt_id = _interrupt_id_from_approval_id(approval_id, pending_ids)
        if interrupt_id not in pending_ids:
            continue
        if interrupt_id in function_output_ids:
            continue
        reason = item.get("reason")
        if isinstance(reason, str) and reason:
            return f"Interrupt '{approval_id}' was rejected by the client: {reason}"
        return f"Interrupt '{approval_id}' was rejected by the client."
    return None


def _approval_id_suffix(interrupt_id: str) -> str:
    return hashlib.sha256(interrupt_id.encode()).hexdigest()[:32]


def _approval_request_id(generated_id: str, interrupt_id: str) -> str:
    """Keep the generated response partition while encoding the interrupt."""
    return f"{generated_id[:-32]}{_approval_id_suffix(interrupt_id)}"


def _interrupt_id_from_approval_id(
    approval_id: str,
    pending_ids: Iterable[str],
) -> str:
    if approval_id in pending_ids:
        return approval_id
    for interrupt_id in pending_ids:
        if approval_id.endswith(_approval_id_suffix(interrupt_id)):
            return interrupt_id
    return approval_id


def _warn_if_competing_approval(items: Sequence[Any], call_id: str) -> None:
    """Log a warning when both shapes target the same interrupt id.

    Specifically: a request containing both a ``function_call_output``
    and an ``mcp_approval_response`` for the same interrupt. Approval request
    ids may carry the encoded ``mcpr_*`` wire form. The
    ``function_call_output`` wins; this helper just surfaces the conflict so
    clients learn the deterministic rule.
    """
    for item in items:
        if (
            _is_mcp_approval_response(item)
            and _interrupt_id_from_approval_id(item["approval_request_id"], (call_id,))
            == call_id
        ):
            logger.warning(
                "Both function_call_output and mcp_approval_response target "
                "interrupt id %r; function_call_output wins.",
                call_id,
            )
            return


def _decode_command(output: Any) -> tuple[Command, bool] | None:
    """Decode a ``function_call_output.output`` payload into a ``Command``."""
    if output is None:
        return None
    if isinstance(output, str):
        text = output.strip()
        if not text:
            return None
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            # Plain string: behave like Command(resume=output).
            return Command(resume=output), True
        return _command_from_object(decoded, raw_string=output)
    if isinstance(output, list):
        # ``output`` can also be a list of content parts; flatten to text.
        text_parts = [
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in output
        ]
        joined = "".join(p for p in text_parts if p)
        if not joined:
            return None
        return _decode_command(joined)
    if isinstance(output, dict):
        return _command_from_object(output)
    return None


def _command_from_object(
    obj: Any, *, raw_string: str | None = None
) -> tuple[Command, bool]:
    """Build a :class:`Command` from a decoded JSON value."""
    if isinstance(obj, dict) and ("resume" in obj or "update" in obj or "goto" in obj):
        return (
            Command(
                resume=obj.get("resume"),
                update=obj.get("update"),
                goto=obj.get("goto") or (),
            ),
            "resume" in obj,
        )
    # JSON didn't look like a Command envelope — treat the whole value
    # (or its original string) as the resume payload.
    return Command(resume=raw_string if raw_string is not None else obj), True


def _merge_command_updates(commands: Sequence[Command]) -> Any | None:
    """Preserve every state write carried by parallel resume commands."""
    updates = [command.update for command in commands if command.update is not None]
    if not updates:
        return None
    if len(updates) == 1:
        return updates[0]

    merged: list[tuple[str, Any]] = []
    for update in updates:
        if isinstance(update, dict):
            merged.extend(update.items())
        elif isinstance(update, (list, tuple)) and all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in update
        ):
            merged.extend(update)
        else:
            merged.append(("__root__", update))
    return merged


def _merge_command_gotos(commands: Sequence[Command]) -> Any:
    """Preserve every destination carried by parallel resume commands."""
    gotos = [command.goto for command in commands if command.goto]
    if not gotos:
        return ()
    if len(gotos) == 1:
        return gotos[0]

    merged: list[Any] = []
    for goto in gotos:
        if isinstance(goto, (list, tuple)):
            merged.extend(goto)
        else:
            merged.append(goto)
    return tuple(merged)


async def emit_interrupts(
    interrupts: Iterable[Interrupt],
    stream: ResponseEventStream,
) -> AsyncIterator[Any]:
    """Yield Responses API events that surface pending interrupts.

    Each interrupt produces *two* output items in the same response:

    1. A ``function_call`` item (name :data:`HITL_FUNCTION_NAME`,
       ``call_id`` = ``interrupt.id``, ``arguments`` = the JSON envelope
       from :func:`interrupt_arguments_json`).
     2. An ``mcp_approval_request`` item with a generated ``mcpr_*`` id,
         ``server_label`` = :data:`HITL_MCP_SERVER_LABEL`, same ``name`` and
         ``arguments``).

    Both items carry the same ``interrupt.id`` so the inbound resume
    matches the same logical pause regardless of which channel the
    client chose.

    Args:
        interrupts: The interrupts to emit (typically from
            :func:`detect_pending_interrupts`).
        stream: The :class:`ResponseEventStream` to emit through.

    Yields:
        Responses API event payload dicts.
    """
    for interrupt in interrupts:
        if not isinstance(interrupt, Interrupt):
            continue
        arguments_json = interrupt_arguments_json(interrupt)

        # Channel 1 — function_call.
        fn = stream.add_output_item_function_call(HITL_FUNCTION_NAME, interrupt.id)
        yield fn.emit_added()
        if arguments_json:
            yield fn.emit_arguments_delta(arguments_json)
        yield fn.emit_arguments_done(arguments_json)
        yield fn.emit_done()

        # Channel 2 — mcp_approval_request with a storage-compatible id.
        approval_builder = stream.add_output_item_mcp_approval_request()
        approval_item = OutputItemMcpApprovalRequest(
            type="mcp_approval_request",
            id=_approval_request_id(approval_builder.item_id, interrupt.id),
            server_label=HITL_MCP_SERVER_LABEL,
            name=HITL_FUNCTION_NAME,
            arguments=arguments_json,
        )
        yield approval_builder.emit_added(approval_item)
        yield approval_builder.emit_done(approval_item)
