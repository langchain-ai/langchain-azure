# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Internal converter package.

Provides request/stream/final translation between LangGraph and the Azure
AI Responses / Invocations APIs.
"""

from ._final import state_to_events
from ._request import (
    build_messages_input,
    build_messages_input_from_text,
    items_to_messages,
)
from ._stream import stream_graph_to_events
from ._utils import extract_text, is_messages_state_schema, last_ai_message_text

__all__ = [
    "build_messages_input",
    "build_messages_input_from_text",
    "items_to_messages",
    "stream_graph_to_events",
    "state_to_events",
    "extract_text",
    "is_messages_state_schema",
    "last_ai_message_text",
]
