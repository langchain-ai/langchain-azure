# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""User-message values used by test-only crash injection."""

from __future__ import annotations

import json
from typing import Any

KEY = "crash"
CRASH_EXIT_CODE = 86
BEFORE_RESPONSE_CREATED = "before_response_created"
AFTER_RESPONSE_CREATED_BEFORE_FIRST_CHECKPOINT = (
    "after_response_created_before_first_checkpoint"
)
AFTER_1PLAN_RESPONSES_CHECKPOINT = "after_1plan_responses_checkpoint"
AFTER_2RESEARCH_CHECKPOINT_BEFORE_METADATA = (
    "after_2research_checkpoint_before_metadata"
)
AFTER_2RESEARCH_GRAPH_CHECKPOINT_BEFORE_METADATA = (
    "after_2research_graph_checkpoint_before_metadata"
)
AFTER_FINAL_RESPONSES_CHECKPOINT_BEFORE_STATE_STORE = (
    "after_final_responses_checkpoint_before_state_store"
)
AFTER_STATE_STORE_BEFORE_COMPLETED = "after_state_store_before_completed"
AFTER_RESPONSE_COMPLETED = "after_response_completed"
NEW_CONVERSATION_FIRST_CHECKPOINT_BEFORE_METADATA = (
    "new_conversation_first_checkpoint_before_metadata"
)
SECOND_TURN_FIRST_CHECKPOINT_BEFORE_METADATA = (
    "second_turn_first_checkpoint_before_metadata"
)
NO_CRASH = "none"

CRASH_POINTS = frozenset(
    {
        BEFORE_RESPONSE_CREATED,
        AFTER_RESPONSE_CREATED_BEFORE_FIRST_CHECKPOINT,
        AFTER_1PLAN_RESPONSES_CHECKPOINT,
        AFTER_2RESEARCH_CHECKPOINT_BEFORE_METADATA,
        AFTER_2RESEARCH_GRAPH_CHECKPOINT_BEFORE_METADATA,
        AFTER_FINAL_RESPONSES_CHECKPOINT_BEFORE_STATE_STORE,
        AFTER_STATE_STORE_BEFORE_COMPLETED,
        AFTER_RESPONSE_COMPLETED,
        NEW_CONVERSATION_FIRST_CHECKPOINT_BEFORE_METADATA,
        SECOND_TURN_FIRST_CHECKPOINT_BEFORE_METADATA,
    }
)
VALID_TRIGGERS = CRASH_POINTS | {NO_CRASH}
INPUT_INSTRUCTION = (
    "Send a JSON object as the user message, for example: "
    '{"crash":"after_response_created_before_first_checkpoint"}. '
    f"Allowed crash values: {', '.join(sorted(VALID_TRIGGERS))}."
)


def parse_crash_point(text: Any) -> str | None:
    """Return a supported crash point from JSON user-message text."""

    if not isinstance(text, str):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get(KEY)
    return value if isinstance(value, str) and value in VALID_TRIGGERS else None


def request_crash_point(request: Any) -> str | None:
    """Read crash-point JSON from a Responses string or user-message input."""

    if not isinstance(request, dict):
        return None
    request_input = request.get("input")
    if isinstance(request_input, str):
        return parse_crash_point(request_input)
    if not isinstance(request_input, list):
        return None

    for item in reversed(request_input):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return parse_crash_point(content)
        if not isinstance(content, list):
            return None
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "input_text"
            and isinstance(part.get("text"), str)
        )
        return parse_crash_point(text)
    return None
