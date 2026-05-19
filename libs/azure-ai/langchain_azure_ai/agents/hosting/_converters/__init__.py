# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Internal converter package.

Provides request/stream/final translation between LangGraph and the Azure
AI Responses / Invocations APIs.
"""

from ._final import state_to_events
from ._hitl import (
    HITL_FUNCTION_NAME,
    HITL_MCP_SERVER_LABEL,
    detect_approval_rejection,
    detect_pending_interrupts,
    emit_interrupts,
    interrupt_arguments_json,
    parse_resume_command,
)
from ._request import (
    build_messages_input,
    build_messages_input_from_text,
    items_to_messages,
)
from ._stream import stream_graph_to_events
from ._utils import extract_text, is_messages_state_schema, last_ai_message_text

__all__ = [
    "HITL_FUNCTION_NAME",
    "HITL_MCP_SERVER_LABEL",
    "build_messages_input",
    "build_messages_input_from_text",
    "detect_approval_rejection",
    "detect_pending_interrupts",
    "emit_interrupts",
    "extract_text",
    "interrupt_arguments_json",
    "is_messages_state_schema",
    "items_to_messages",
    "last_ai_message_text",
    "parse_resume_command",
    "state_to_events",
    "stream_graph_to_events",
]
