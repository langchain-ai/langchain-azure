# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Human-in-the-loop translation between LangGraph and the Responses API.

LangGraph pauses execution when a node calls ``langgraph.types.interrupt``.
The pause is checkpointed and surfaced on
:attr:`langgraph.types.StateSnapshot.interrupts`. Resume happens by
invoking the graph again with a :class:`langgraph.types.Command` carrying
``resume`` / ``update`` / ``goto`` fields.

We map this onto the OpenAI Responses API using only the standard item
types, so off-the-shelf Responses clients work without modification:

* Each pending interrupt is emitted as a ``function_call`` output item
  with name :data:`LANGGRAPH_INTERRUPT_NAME` and ``call_id`` equal to the
  LangGraph interrupt id. Its ``arguments`` field carries the interrupt
  value (JSON-encoded when it isn't already a string).
* The client resumes by posting a matching ``function_call_output`` input
  item (same ``call_id``). Its ``output`` is a JSON object with any of
  ``{"resume", "update", "goto"}``. A non-JSON output string is treated
  as the resume value verbatim — matching the "ask a question, get a
  string back" pattern used by the most common HITL samples.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Final

from azure.ai.agentserver.responses import ResponseEventStream
from azure.ai.agentserver.responses.models import FunctionCallOutputItemParam
from langgraph.types import Command, Interrupt

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import StateSnapshot

logger = logging.getLogger(__name__)

HITL_FUNCTION_NAME: Final[str] = "__hosted_agent_adapter_interrupt__"
"""Reserved ``function_call.name`` used to surface a LangGraph interrupt.

The string value matches the ``HUMAN_IN_THE_LOOP_FUNCTION_NAME`` used by
``azure-ai-agentserver-langgraph`` so clients can share the same
discriminator across both hosts. Treat the literal as opaque and match
on it via this symbol.
"""


async def detect_pending_interrupts(
    graph: "CompiledStateGraph", config: dict[str, Any]
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
    """Render an interrupt's value as the ``arguments`` JSON string.

    Strings pass through unchanged; everything else is JSON-encoded with
    a string fallback when the value is not JSON-serializable.

    Args:
        interrupt: The interrupt to encode.

    Returns:
        The string to use as ``function_call.arguments``.
    """
    value = interrupt.value
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        logger.warning(
            "Interrupt value not JSON-serializable; falling back to str()."
        )
        return json.dumps(str(value))


def parse_resume_command(
    items: Sequence[Any],
    pending: Sequence[Interrupt],
) -> tuple[Command | None, frozenset[str]]:
    """Build a resume :class:`Command` from request input items, if present.

    Looks for ``FunctionCallOutputItemParam`` items whose ``call_id``
    matches one of the pending interrupts. The matched item's ``output``
    field is decoded:

    * JSON object — fields ``resume`` / ``update`` / ``goto`` populate the
      :class:`Command`.
    * Anything else (string, malformed JSON, list) — treated as the raw
      resume value.

    When multiple resume items are present we currently honour the first
    one matching a pending interrupt and ignore the rest; consumed call
    ids are returned so callers can filter them out of the regular
    message-construction path.

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
    for item in items:
        if not isinstance(item, FunctionCallOutputItemParam):
            continue
        call_id = item.call_id
        if call_id not in pending_by_id:
            continue
        command = _decode_command(item.output)
        if command is None:
            continue
        return command, frozenset({call_id})

    return None, frozenset()


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
    if isinstance(obj, dict) and (
        "resume" in obj or "update" in obj or "goto" in obj
    ):
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
) -> AsyncIterator[dict[str, Any]]:
    """Yield Responses API events that surface pending interrupts.

    Each interrupt becomes one ``function_call`` output item named
    :data:`LANGGRAPH_INTERRUPT_NAME` with ``call_id == interrupt.id``.

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
        fn = stream.add_output_item_function_call(
            HITL_FUNCTION_NAME, interrupt.id
        )
        yield fn.emit_added()
        if arguments_json:
            yield fn.emit_arguments_delta(arguments_json)
        yield fn.emit_arguments_done(arguments_json)
        yield fn.emit_done()
