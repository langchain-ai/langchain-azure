"""Graph-aware rejection conversion without executing graph code."""

from typing import Any

import pytest
from langgraph.types import Command

from langchain_azure_ai.agents.hosting._converters import (
    build_langchain_rejection_command,
    collect_approval_rejections,
    interrupt_output_items,
    merge_rejection_command,
)

from .conftest import pending_interrupt


def test_collect_rejections_preserves_reasons_and_function_output_priority() -> None:
    pending = [pending_interrupt(id="first"), pending_interrupt(id="second")]
    approvals = [
        item
        for item in interrupt_output_items(pending)
        if item["type"] == "mcp_approval_request"
    ]
    items: list[dict[str, Any]] = [
        {
            "type": "mcp_approval_response",
            "approval_request_id": approval["id"],
            "approve": False,
            "reason": reason,
        }
        for approval, reason in zip(approvals, ("Not authorized", "No thanks"))
    ]
    assert collect_approval_rejections(items, pending) == {
        "first": "Not authorized",
        "second": "No thanks",
    }
    items.append(
        {"type": "function_call_output", "call_id": "second", "output": "answer"}
    )
    assert collect_approval_rejections(items, pending) == {"first": "Not authorized"}


def test_langchain_rejection_covers_all_actions_and_preserves_reason() -> None:
    pending = pending_interrupt(
        value={
            "action_requests": [
                {"name": name, "args": {}} for name in ("send", "save")
            ],
            "review_configs": [
                {"action_name": name, "allowed_decisions": ["approve", "reject"]}
                for name in ("send", "save")
            ],
        }
    )
    command = build_langchain_rejection_command({pending.id: "Denied"}, [pending])
    assert command is not None
    assert command.resume == {
        pending.id: {
            "decisions": [
                {"type": "reject", "message": "Denied"},
                {"type": "reject", "message": "Denied"},
            ]
        }
    }


@pytest.mark.parametrize(
    "value",
    [
        "question?",
        {"question": "Approve?"},
        {"action_requests": [], "review_configs": []},
        {
            "action_requests": [{"name": "send", "args": {}}],
            "review_configs": [
                {"action_name": "send", "allowed_decisions": ["approve"]}
            ],
        },
        {
            "action_requests": [{"name": "send", "args": {}}],
            "review_configs": [
                {"action_name": "other", "allowed_decisions": ["reject"]}
            ],
        },
    ],
)
def test_unknown_or_disallowed_rejection_is_not_guessed(value: Any) -> None:
    pending = pending_interrupt(value=value)
    assert build_langchain_rejection_command({pending.id: None}, [pending]) is None


def test_parallel_rejection_merges_sibling_answer_and_state_updates() -> None:
    pending = [pending_interrupt(id="first"), pending_interrupt(id="second")]
    command = merge_rejection_command(
        Command(resume={"first": {"approved": False}}, update={"reason": "Denied"}),
        Command(resume={"second": "answer"}, update={"result": 1}, goto="next"),
        pending,
    )
    assert command.resume == {"first": {"approved": False}, "second": "answer"}
    assert command.update == [("result", 1), ("reason", "Denied")]
    assert command.goto == "next"
