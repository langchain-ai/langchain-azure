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
  is surfaced to the host as a rejection signal). For a recognized LangChain
  ``HumanInTheLoopMiddleware`` payload, the boolean choice is instead expanded
  into one ordered decision per action.

The MCP approval shortcut recognizes ``HumanInTheLoopMiddleware`` requests by
their complete ``action_requests`` / ``review_configs`` structure. That shape is
therefore reserved on this channel. A custom interrupt using the same shape must
use ``function_call_output`` when its value must be preserved verbatim.

When both shapes target the same ``interrupt.id`` in one request,
``function_call_output`` wins (it carries the richer payload) and a
warning is logged. Conflicting MCP approval responses for the same interrupt
fail the request without resuming the graph.

Each continuation must answer the entire current batch before graph execution.
Missing results or approval decisions fail validation without saving partial
answers. ``update`` / ``goto`` fields may accompany a resume value, but cannot
replace it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import replace
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

from .._messages import MISSING_FUNCTION_OUTPUT, MISSING_MCP_APPROVAL

if TYPE_CHECKING:
    from langgraph.types import StateSnapshot

logger = logging.getLogger(__name__)
_HITL_DECISIONS: Final = frozenset({"approve", "edit", "reject", "respond"})

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


class IncompleteResumeError(ValueError):
    """A continuation that does not answer every pending interrupt.

    Args:
        message: Client-facing validation message.
        param: Invalid request parameter, or ``None`` for MCP approvals.
    """

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param

    def as_error(self) -> dict[str, Any]:
        """Return the Responses-compatible error object."""
        return {
            "message": str(self),
            "type": "invalid_request_error",
            "param": self.param,
            "code": None,
        }


async def detect_pending_interrupts(
    graph: Runnable[Any, Any], config: RunnableConfig
) -> tuple[Interrupt, ...]:
    """Return the interrupts pending on the checkpointed state, if any.

    ``StateSnapshot.interrupts`` accumulates every interrupt recorded on the
    checkpoint and is *not* pruned as they are answered, so after a partial
    resume of parallel interrupts it may still report the ones already
    satisfied. Deriving the list from ``StateSnapshot.tasks`` and skipping
    tasks that produced node output keeps the initial lookup aligned with
    what is outstanding in the common case. A node pausing again after a
    resume can have an empty result dictionary. If no result-less tasks
    remain, only these empty dictionaries are considered, not completed
    siblings with other results.

    This remains a heuristic: both a node pausing again and a completed
    sibling returning no state-channel updates can have an empty result.
    The completed sibling's retained interrupt can therefore be mistaken
    for a pending one, causing complete-batch validation to reject a valid
    continuation with HTTP 400. Reproduced with LangGraph 1.2.12; see
    https://github.com/langchain-ai/langchain-azure/pull/1180.

    After a graph invocation, :func:`track_pending_interrupts` provides the
    authoritative active set.

    Args:
        graph: The runnable to inspect. Runnables without an ``aget_state``
            method have no checkpointed interrupts and return an empty tuple.
        config: The :class:`RunnableConfig` identifying the thread.

    Returns:
        A tuple of :class:`Interrupt` objects (empty when none pending or
        when the graph has no checkpointer attached).

    Raises:
        Exception: If reading checkpoint state fails. A failed lookup must not
            bypass continuation validation.
    """
    get_state = getattr(graph, "aget_state", None)
    checkpointer = getattr(graph, "checkpointer", True)
    if get_state is None or checkpointer is None or checkpointer is False:
        return ()
    snapshot: "StateSnapshot | None" = await get_state(config)
    if snapshot is None:
        return ()
    tasks = tuple(getattr(snapshot, "tasks", None) or ())
    pending_tasks = tuple(
        task for task in tasks if getattr(task, "result", None) is None
    )
    if not pending_tasks:
        pending_tasks = tuple(
            task
            for task in tasks
            if isinstance(result := getattr(task, "result", None), dict) and not result
        )

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


def _function_resume_commands(
    items: Sequence[Any],
    pending_ids: Iterable[str],
) -> dict[str, tuple[Command, bool]]:
    pending_id_set = set(pending_ids)
    commands: dict[str, tuple[Command, bool]] = {}
    for item in items:
        if not _is_function_call_output(item):
            continue
        call_id = item["call_id"]
        if call_id not in pending_id_set or call_id in commands:
            continue
        decoded = _decode_command(item["output"])
        if decoded is not None:
            commands[call_id] = decoded
    return commands


def validate_resume_inputs(
    items: Sequence[Any],
    pending: Sequence[Interrupt],
    *,
    output_items: Sequence[Any] = (),
) -> None:
    """Require a complete batch of valid responses before resuming.

    Inputs are normalized before validation. Approvals and rejections both
    answer an MCP request. Rich function outputs must carry a resume value,
    not only ``update`` or ``goto``. Empty and whitespace-only string outputs
    are valid values. Duplicate or unknown IDs cannot satisfy another pending
    interrupt.

    Args:
        items: Current request input items, excluding prior conversation input.
        pending: Interrupts pending on the selected checkpoint.
        output_items: Prior output items used to retain emitted MCP request IDs
            in validation messages. Invocations use deterministic IDs by default.

    Raises:
        IncompleteResumeError: If any pending interrupt is unanswered.
    """
    if not pending:
        return
    command, _ = parse_resume_command(items, pending, require_complete=False)
    validate_resume_command(command, pending, items=items, output_items=output_items)


def _is_langgraph_resume_map(value: Any) -> TypeGuard[dict[str, Any]]:
    """Match LangGraph's ID-map interpretation, not pending-ID membership."""
    return isinstance(value, dict) and all(
        isinstance(key, str) and re.fullmatch(r"[0-9a-f]{32}", key) is not None
        for key in value
    )


def validate_resume_command(
    command: Command | None,
    pending: Sequence[Interrupt],
    *,
    items: Sequence[Any] = (),
    output_items: Sequence[Any] = (),
) -> None:
    """Validate a normalized command and any MCP approval decisions.

    A single pending interrupt accepts a scalar resume value. Dictionaries
    whose keys are all 32-character lowercase hexadecimal interrupt IDs,
    including an empty dictionary, follow LangGraph's ID-map semantics and
    must contain the pending ID. Parallel interrupts require an ID-keyed map.
    Non-ID keys invalidate a parallel resume even when MCP decisions answer
    every pending request.
    Explicit MCP rejections also answer their requests, even when they do not
    produce a resumable command.

    Args:
        command: Command built by the host's existing input/resume hook.
        pending: Interrupts pending on the selected checkpoint.
        items: Current protocol items, used for MCP decisions and error messages.
        output_items: Prior output items retaining emitted MCP request IDs.

    Raises:
        IncompleteResumeError: If pending answers are incomplete or a parallel
            resume is not an ID-keyed map.
    """
    answered: set[str] = set()
    if command is not None and command.resume is not None:
        resume_is_map = _is_langgraph_resume_map(command.resume)
        if len(pending) > 1 and not resume_is_map:
            raise IncompleteResumeError(
                "Parallel interrupts require a resume map keyed only by interrupt IDs.",
                param="input",
            )
        if len(pending) == 1 and not resume_is_map:
            answered.add(pending[0].id)
        elif resume_is_map:
            answered.update(item.id for item in pending if item.id in command.resume)
    _validate_resume_answers(items, pending, answered, output_items)


def _validate_resume_answers(
    items: Sequence[Any],
    pending: Sequence[Interrupt],
    answered: set[str],
    output_items: Sequence[Any],
) -> None:
    pending_ids = dict.fromkeys(interrupt.id for interrupt in pending)
    commands = _function_resume_commands(items, pending_ids)
    for item in items:
        if not _is_mcp_approval_response(item):
            continue
        interrupt_id = _interrupt_id_from_approval_id(
            item["approval_request_id"], pending_ids
        )
        if interrupt_id in pending_ids and interrupt_id not in commands:
            answered.add(interrupt_id)
    missing = [
        interrupt_id for interrupt_id in pending_ids if interrupt_id not in answered
    ]
    if not missing:
        return

    mcp_channel = any(_is_mcp_approval_response(item) for item in items) and not any(
        _is_function_call_output(item) for item in items
    )
    if mcp_channel:
        approval_ids = {
            interrupt_id: f"mcpr_{_approval_id_suffix(interrupt_id)}"
            for interrupt_id in pending_ids
        }
        for item in output_items:
            if (
                isinstance(item, dict)
                and item.get("type") == "mcp_approval_request"
                and item.get("name") == HITL_FUNCTION_NAME
                and isinstance(item.get("id"), str)
            ):
                interrupt_id = _interrupt_id_from_approval_id(item["id"], pending_ids)
                if interrupt_id in pending_ids:
                    approval_ids[interrupt_id] = item["id"]
        raise IncompleteResumeError(
            MISSING_MCP_APPROVAL.format(
                approval_request_ids=", ".join(approval_ids[key] for key in missing)
            )
        )
    raise IncompleteResumeError(
        MISSING_FUNCTION_OUTPUT.format(call_id=missing[0]),
        param="input",
    )


def parse_resume_command(
    items: Sequence[Any],
    pending: Sequence[Interrupt],
    *,
    require_complete: bool = True,
) -> tuple[Command | None, frozenset[str]]:
    """Build a resume :class:`Command` from a complete batch of input items.

    Two input shapes are accepted, both keyed by ``interrupt.id``:

    * :class:`FunctionCallOutputItemParam` matched by ``call_id``. Its
      ``output`` field is decoded — a JSON object with ``resume`` and optional
      ``update`` / ``goto`` fields populates the :class:`Command`;
      anything else (string, malformed JSON, list of content parts) is
      treated as the raw resume value.
    * :class:`MCPApprovalResponse` matched by ``approval_request_id``. For a
      recognized LangChain ``HumanInTheLoopMiddleware`` request, its boolean
      choice becomes one ordered decision per action. Otherwise,
      ``approve=True`` retains the original interrupt-value echo behavior and
      ``approve=False`` is surfaced by :func:`validate_approval_responses`.
      Recognition uses the complete ``action_requests`` / ``review_configs``
      structure, which is reserved for this fast path. Use
      ``function_call_output`` to preserve an identically shaped custom value.

    Conflict resolution: when both shapes target the same interrupt id
    in one request, the ``function_call_output`` wins (richer payload)
    and a warning is logged. This is a deliberate, deterministic
    departure from Agent Framework's order-dependent last-write-wins.
    Before executing the command, hosts must validate batch completeness and
    call :func:`validate_approval_responses` to reject conflicting decisions.

    Resume shape: answers are normalized into ``Command(resume={id: value})``,
    including a single pending interrupt. With a single pending interrupt,
    existing native ID maps in function outputs remain unchanged for
    compatibility. MCP decisions always carry one answer value and are nested
    under their interrupt ID, regardless of their dictionary keys. This
    normalization applies to built-in protocol items, not commands returned
    directly by custom hooks. Every pending interrupt must be answered in the
    same request; partial batches are rejected before any state updates or
    graph execution.

    Args:
        items: Resolved input items from the request.
        pending: Pending interrupts on the graph's checkpointed state.
        require_complete: Reject partial batches by default. Hosts set this to
            ``False`` to normalize input before calling their validation hook.

    Returns:
        A ``(command, consumed_call_ids)`` pair. ``command`` is ``None``
        when the input produces no resumable command.

    Raises:
        IncompleteResumeError: If the request does not answer every pending
            interrupt with a resume value or MCP approval decision.
    """
    if require_complete:
        validate_resume_inputs(items, pending)
    if not pending:
        return None, frozenset()

    pending_by_id: dict[str, Interrupt] = {it.id: it for it in pending}
    # interrupt id -> (decoded command, explicitly carries resume), in
    # first-seen order. ``Command.resume is None`` alone cannot distinguish
    # an omitted resume from an explicit ``{"resume": null}``.
    commands = _function_resume_commands(items, pending_by_id)
    approval_interrupt_ids: set[str] = set()
    # interrupt id -> the wire id that produced it (the encoded ``mcpr_*``
    # id for approvals, the raw interrupt id for function call outputs).
    consumed: dict[str, str] = {interrupt_id: interrupt_id for interrupt_id in commands}

    # Pass 1 — prefer function_call_output (richer payload).
    for call_id in commands:
        _warn_if_competing_approval(items, call_id)

    # Pass 2 — fall back to mcp_approval_response.
    for item in items:
        if not _is_mcp_approval_response(item):
            continue
        approval_id = item["approval_request_id"]
        interrupt_id = _interrupt_id_from_approval_id(approval_id, pending_by_id)
        interrupt_obj = pending_by_id.get(interrupt_id)
        if interrupt_obj is None or interrupt_id in commands:
            continue
        decision_type = "approve" if item["approve"] else "reject"
        hitl_middleware_request = _parse_hitl_middleware_request(interrupt_obj.value)
        if hitl_middleware_request is not None:
            hitl_middleware_response = _build_hitl_middleware_response(
                hitl_middleware_request, decision_type, item.get("reason")
            )
            if hitl_middleware_response is None:
                # At least one action disallows this decision. Leave the
                # interrupt pending so validate_approval_responses() can report it.
                continue
            commands[interrupt_id] = (
                Command(resume=hitl_middleware_response),
                True,
            )
        elif not item["approve"]:
            # Custom interrupt rejection is handled by
            # validate_approval_responses().
            continue
        else:
            commands[interrupt_id] = (Command(resume=interrupt_obj.value), True)
        consumed[interrupt_id] = approval_id
        approval_interrupt_ids.add(interrupt_id)

    if not commands:
        return None, frozenset()

    consumed_ids = frozenset(consumed.values())
    if len(pending_by_id) == 1:
        interrupt_id, (command, has_resume) = next(iter(commands.items()))
        if has_resume and (
            interrupt_id in approval_interrupt_ids
            or not _is_langgraph_resume_map(command.resume)
        ):
            command = replace(command, resume={interrupt_id: command.resume})
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


def validate_approval_responses(
    items: Sequence[Any],
    pending: Sequence[Interrupt],
) -> str | None:
    """Return a validation error when an approval shortcut cannot be applied.

    Scans for :class:`MCPApprovalResponse` items whose
    ``approval_request_id`` matches a pending interrupt. Recognized LangChain
    ``HumanInTheLoopMiddleware`` requests are valid only when every action
    allows the selected decision. For other interrupt payloads,
    ``approve=False`` returns an error instead of resuming the graph.

    The host's :meth:`handle_create` calls this *before* attempting to
    resume so a rejection short-circuits the turn into
    ``response.failed`` instead of being silently dropped.

    Args:
        items: Resolved input items from the request.
        pending: Pending interrupts on the graph's checkpointed state.

    Returns:
        The validation error, or ``None`` when every approval response is valid.
    """
    if not pending:
        return None
    pending_by_id = {interrupt.id: interrupt for interrupt in pending}
    pending_ids = set(pending_by_id)
    function_output_ids = {
        item["call_id"]
        for item in items
        if _is_function_call_output(item)
        and item["call_id"] in pending_ids
        and _decode_command(item["output"]) is not None
    }
    conflicting_approval_ids = _conflicting_approval_interrupt_ids(
        items, pending_ids, function_output_ids
    )
    if conflicting_approval_ids:
        ordered_ids = [
            interrupt.id
            for interrupt in pending
            if interrupt.id in conflicting_approval_ids
        ]
        return (
            f"Interrupts {ordered_ids!r} received conflicting MCP approval responses."
        )

    for item in items:
        if not _is_mcp_approval_response(item):
            continue
        approval_id = item["approval_request_id"]
        interrupt_id = _interrupt_id_from_approval_id(approval_id, pending_ids)
        if interrupt_id not in pending_ids:
            continue
        if interrupt_id in function_output_ids:
            continue
        interrupt_obj = pending_by_id[interrupt_id]
        decision_type = "approve" if item["approve"] else "reject"
        hitl_middleware_request = _parse_hitl_middleware_request(interrupt_obj.value)
        if hitl_middleware_request is not None:
            disallowed_action = _find_disallowed_hitl_middleware_action(
                hitl_middleware_request, decision_type
            )
            if disallowed_action is None:
                continue
            return (
                f"Interrupt '{approval_id}' cannot {decision_type} action "
                f"'{disallowed_action['name']}': decision is not allowed."
            )
        if item["approve"]:
            continue
        reason = item.get("reason")
        if isinstance(reason, str) and reason:
            return f"Interrupt '{approval_id}' was rejected by the client: {reason}"
        return f"Interrupt '{approval_id}' was rejected by the client."
    return None


def _conflicting_approval_interrupt_ids(
    items: Sequence[Any],
    pending_ids: Iterable[str],
    function_output_ids: Iterable[str],
) -> frozenset[str]:
    """Return interrupts with both approve and reject responses."""
    pending_id_set = set(pending_ids)
    function_output_id_set = set(function_output_ids)
    decisions: dict[str, bool] = {}
    conflicts: set[str] = set()
    for item in items:
        if not _is_mcp_approval_response(item):
            continue
        interrupt_id = _interrupt_id_from_approval_id(
            item["approval_request_id"], pending_id_set
        )
        if interrupt_id not in pending_id_set or interrupt_id in function_output_id_set:
            continue
        if interrupt_id in decisions and decisions[interrupt_id] != item["approve"]:
            conflicts.add(interrupt_id)
        decisions[interrupt_id] = item["approve"]
    return frozenset(conflicts)


def _parse_hitl_middleware_request(
    value: Any,
) -> dict[str, list[dict[str, Any]]] | None:
    """Return a validated HumanInTheLoopMiddleware request."""
    if not isinstance(value, dict):
        return None
    actions = value.get("action_requests")
    reviews = value.get("review_configs")
    if (
        not isinstance(actions, list)
        or not actions
        or not isinstance(reviews, list)
        or len(actions) != len(reviews)
    ):
        return None
    for action, review in zip(actions, reviews, strict=True):
        allowed = review.get("allowed_decisions") if isinstance(review, dict) else None
        if (
            not isinstance(action, dict)
            or not isinstance(action.get("name"), str)
            or not action["name"]
            or not isinstance(action.get("args"), dict)
            or not isinstance(review, dict)
            or review.get("action_name") != action["name"]
            or not isinstance(allowed, list)
            or not allowed
            or not all(
                isinstance(decision, str) and decision in _HITL_DECISIONS
                for decision in allowed
            )
        ):
            return None
    return {"action_requests": actions, "review_configs": reviews}


def _find_disallowed_hitl_middleware_action(
    hitl_middleware_request: dict[str, list[dict[str, Any]]],
    decision_type: str,
) -> dict[str, Any] | None:
    """Return the first action that disallows the decision."""
    actions = hitl_middleware_request["action_requests"]
    reviews = hitl_middleware_request["review_configs"]
    for action, review in zip(actions, reviews, strict=True):
        if decision_type not in review["allowed_decisions"]:
            return action
    return None


def _build_hitl_middleware_response(
    hitl_middleware_request: dict[str, list[dict[str, Any]]],
    decision_type: str,
    message: Any = None,
) -> dict[str, Any] | None:
    """Build the all-or-nothing HumanInTheLoopMiddleware response."""
    actions = hitl_middleware_request["action_requests"]
    if (
        _find_disallowed_hitl_middleware_action(hitl_middleware_request, decision_type)
        is not None
    ):
        return None

    decision = {"type": decision_type}
    if decision_type == "reject" and isinstance(message, str) and message:
        decision["message"] = message
    return {"decisions": [dict(decision) for _ in actions]}


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
