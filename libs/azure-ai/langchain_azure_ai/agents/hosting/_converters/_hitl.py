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
2. An ``mcp_approval_request`` output item with ``id == interrupt.id``,
   ``server_label == "langgraph"``, the same ``name``, and the same
   ``arguments`` envelope.

Both items reference the *same* ``interrupt.id``. The client resumes by
posting either:

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

import json
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Final

from azure.ai.agentserver.responses import ResponseEventStream
from azure.ai.agentserver.responses.models import (
    FunctionCallOutputItemParam,
    MCPApprovalResponse,
)
from azure.ai.agentserver.responses.models._generated import (
    OutputItemMcpApprovalRequest,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
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

HITL_MCP_SERVER_LABEL: Final[str] = "langgraph"
"""``server_label`` stamped on the ``mcp_approval_request`` we emit.

We borrow the MCP approval item type as a generic approval channel
(mirroring Microsoft Agent Framework's `foundry_hosting`). The label
exists so clients can discriminate our HITL items from real MCP
approval requests at a glance.
"""


async def detect_pending_interrupts(
    graph: "CompiledStateGraph", config: RunnableConfig
) -> tuple[Interrupt, ...]:
    """Return the interrupts pending on the checkpointed state, if any.

    Args:
        graph: The compiled state graph to inspect.
        config: The :class:`RunnableConfig` identifying the thread.

    Returns:
        A tuple of :class:`Interrupt` objects (empty when none pending or
        when the graph has no checkpointer attached).
    """
    try:
        snapshot: "StateSnapshot | None" = await graph.aget_state(config)
    except Exception:  # noqa: BLE001
        # No checkpointer / unknown thread / provider error — treat as
        # "nothing pending" and let the regular path run.
        logger.debug("aget_state failed; assuming no pending interrupts", exc_info=True)
        return ()
    if snapshot is None:
        return ()
    interrupts = getattr(snapshot, "interrupts", None) or ()
    return tuple(it for it in interrupts if isinstance(it, Interrupt))


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

    # Pass 1 — prefer function_call_output (richer payload).
    for item in items:
        if not isinstance(item, FunctionCallOutputItemParam):
            continue
        call_id = item.call_id
        if call_id not in pending_by_id:
            continue
        command = _decode_command(item.output)
        if command is None:
            continue
        _warn_if_competing_approval(items, call_id)
        return command, frozenset({call_id})

    # Pass 2 — fall back to mcp_approval_response (approve-only).
    for item in items:
        if not isinstance(item, MCPApprovalResponse):
            continue
        approval_id = item.approval_request_id
        interrupt_obj = pending_by_id.get(approval_id)
        if interrupt_obj is None:
            continue
        if not item.approve:
            # Rejection is surfaced via ``detect_approval_rejection``
            # rather than as a ``Command``. Skip it here.
            continue
        return Command(resume=interrupt_obj.value), frozenset({approval_id})

    return None, frozenset()


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
    for item in items:
        if not isinstance(item, MCPApprovalResponse):
            continue
        if item.approve:
            continue
        approval_id = item.approval_request_id
        if approval_id not in pending_ids:
            continue
        reason = getattr(item, "reason", None)
        if isinstance(reason, str) and reason:
            return f"Interrupt '{approval_id}' was rejected by the client: {reason}"
        return f"Interrupt '{approval_id}' was rejected by the client."
    return None


def _warn_if_competing_approval(items: Sequence[Any], call_id: str) -> None:
    """Log a warning when both shapes target the same interrupt id.

    Specifically: a request containing both a ``function_call_output``
    *and* an ``mcp_approval_response`` keyed by the same ``call_id``.
    The ``function_call_output`` wins; this helper just surfaces the
    conflict so clients learn the deterministic rule.
    """
    for item in items:
        if (
            isinstance(item, MCPApprovalResponse)
            and item.approval_request_id == call_id
        ):
            logger.warning(
                "Both function_call_output and mcp_approval_response target "
                "interrupt id %r; function_call_output wins.",
                call_id,
            )
            return


def _decode_command(output: Any) -> Command | None:
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
            return Command(resume=output)
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


def _command_from_object(obj: Any, *, raw_string: str | None = None) -> Command | None:
    """Build a :class:`Command` from a decoded JSON value."""
    if isinstance(obj, dict) and ("resume" in obj or "update" in obj or "goto" in obj):
        return Command(
            resume=obj.get("resume"),
            update=obj.get("update"),
            goto=obj.get("goto") or (),
        )
    # JSON didn't look like a Command envelope — treat the whole value
    # (or its original string) as the resume payload.
    return Command(resume=raw_string if raw_string is not None else obj)


async def emit_interrupts(
    interrupts: Iterable[Interrupt],
    stream: ResponseEventStream,
) -> AsyncIterator[Any]:
    """Yield Responses API events that surface pending interrupts.

    Each interrupt produces *two* output items in the same response:

    1. A ``function_call`` item (name :data:`HITL_FUNCTION_NAME`,
       ``call_id`` = ``interrupt.id``, ``arguments`` = the JSON envelope
       from :func:`interrupt_arguments_json`).
    2. An ``mcp_approval_request`` item (``id`` = ``interrupt.id``,
       ``server_label`` = :data:`HITL_MCP_SERVER_LABEL`, same ``name``
       and ``arguments``).

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

        # Channel 2 — mcp_approval_request with caller-supplied id.
        #
        # The high-level ``output_item_mcp_approval_request`` helper
        # auto-generates an ``mcpr_*`` id we can't override; we go
        # through the low-level builder so the wire id equals the
        # interrupt id. The builder still allocates its own ``mcpr_*``
        # internal item_id (used only by item-level state events, which
        # aren't emitted for simple items) — that's harmless.
        approval_builder = stream.add_output_item_mcp_approval_request()
        approval_item = OutputItemMcpApprovalRequest(
            id=interrupt.id,
            server_label=HITL_MCP_SERVER_LABEL,
            name=HITL_FUNCTION_NAME,
            arguments=arguments_json,
        )
        yield approval_builder.emit_added(approval_item)
        yield approval_builder.emit_done(approval_item)
