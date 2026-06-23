"""Sample: verify that dynamic-sessions tools delete sessions correctly with create_agent.

This sample demonstrates and tests the ``delete_session_after_invocation`` feature
introduced in https://github.com/langchain-ai/langchain-azure/pull/715.

When ``delete_session_after_invocation=True``, the tool must call
``delete_session()`` exactly once after every agent tool-call, preventing the
session pool from exhausting its slot limit.

The sample covers:

1. **SessionsPythonREPLTool** — session deleted after a Python code execution.
2. **SessionsBashTool** — session deleted after a bash command execution.
3. **No deletion by default** — when the flag is ``False`` (or omitted),
   ``delete_session()`` is never called.
4. **Multiple invocations** — each agent turn that triggers a tool call
   results in exactly one ``delete_session()`` call.

Usage (no API keys required — all HTTP calls are mocked)::

    python samples/dynamic-sessions/test_session_deletion.py

All assertions are run in-process; an ``AssertionError`` is raised on failure.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List
from unittest import mock

from azure.core.credentials import AccessToken
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from langchain_azure_dynamic_sessions import SessionsBashTool, SessionsPythonREPLTool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POOL_MANAGEMENT_ENDPOINT = (
    "https://westus2.dynamicsessions.io/subscriptions/00000000-0000-0000-0000-000000000000"
    "/resourceGroups/sessions-rg/sessionPools/my-pool"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_python_execution_response(result: Any = 4, stdout: str = "") -> Dict[str, Any]:
    """Return a fake successful Python execution API response."""
    return {
        "$id": "1",
        "properties": {
            "$id": "2",
            "status": "Success",
            "stdout": stdout,
            "stderr": "",
            "result": result,
            "executionTimeInMilliseconds": 10,
        },
    }


def _make_bash_execution_response(stdout: str = "", exit_code: int = 0) -> Dict[str, Any]:
    """Return a fake successful bash execution API response."""
    return {
        "identifier": "test-session",
        "status": str(exit_code),
        "result": {
            "stdout": stdout,
            "stderr": "",
            "executionTimeInMilliseconds": 10,
        },
    }


def _make_tool_call_ai_message(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Return the JSON representation of an AIMessage with a tool call.

    ``FakeListChatModel`` accepts plain strings, so we serialise the
    ``AIMessage`` content ourselves via its ``model_dump`` representation.
    """
    msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": tool_name, "args": tool_args}],
    )
    # FakeListChatModel returns the string as-is; we instead pass the object
    # directly via the ``responses`` list.
    return msg  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_python_repl_session_deleted_after_agent_tool_call() -> None:
    """Agent using SessionsPythonREPLTool(delete_session_after_invocation=True)
    must call delete_session() once per tool invocation.
    """
    print("test_python_repl_session_deleted_after_agent_tool_call ... ", end="", flush=True)

    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    # The fake model first emits a tool-call, then a plain text finish.
    fake_model = FakeListChatModel(
        responses=[
            _make_tool_call_ai_message(
                tool.name, {"__arg1": "print(2 + 2)"}
            ),
            AIMessage(content="The answer is 4."),
        ]
    )
    agent = create_agent(model=fake_model, tools=[tool])

    with (
        mock.patch(
            "azure.identity.DefaultAzureCredential.get_token",
            return_value=AccessToken("token_value", int(time.time() + 1000)),
        ),
        mock.patch(
            "requests.post",
            return_value=mock.MagicMock(
                json=mock.MagicMock(
                    return_value=_make_python_execution_response(result=4, stdout="4\n")
                )
            ),
        ),
        mock.patch.object(
            SessionsPythonREPLTool, "delete_session", autospec=True
        ) as mock_delete_session,
    ):
        agent.invoke({"messages": [{"role": "user", "content": "What is 2 + 2?"}]})

    mock_delete_session.assert_called_once_with(tool)
    print("OK")


def test_bash_session_deleted_after_agent_tool_call() -> None:
    """Agent using SessionsBashTool(delete_session_after_invocation=True)
    must call delete_session() once per tool invocation.
    """
    print("test_bash_session_deleted_after_agent_tool_call ... ", end="", flush=True)

    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    fake_model = FakeListChatModel(
        responses=[
            _make_tool_call_ai_message(tool.name, {"__arg1": "echo hello"}),
            AIMessage(content="Done."),
        ]
    )
    agent = create_agent(model=fake_model, tools=[tool])

    with (
        mock.patch(
            "azure.identity.DefaultAzureCredential.get_token",
            return_value=AccessToken("token_value", int(time.time() + 1000)),
        ),
        mock.patch(
            "requests.post",
            return_value=mock.MagicMock(
                json=mock.MagicMock(
                    return_value=_make_bash_execution_response(stdout="hello\n")
                )
            ),
        ),
        mock.patch.object(
            SessionsBashTool, "delete_session", autospec=True
        ) as mock_delete_session,
    ):
        agent.invoke({"messages": [{"role": "user", "content": "Run: echo hello"}]})

    mock_delete_session.assert_called_once_with(tool)
    print("OK")


def test_session_not_deleted_by_default() -> None:
    """When delete_session_after_invocation is not set (default False),
    delete_session() must NOT be called.
    """
    print("test_session_not_deleted_by_default ... ", end="", flush=True)

    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)

    fake_model = FakeListChatModel(
        responses=[
            _make_tool_call_ai_message(tool.name, {"__arg1": "1 + 1"}),
            AIMessage(content="The result is 2."),
        ]
    )
    agent = create_agent(model=fake_model, tools=[tool])

    with (
        mock.patch(
            "azure.identity.DefaultAzureCredential.get_token",
            return_value=AccessToken("token_value", int(time.time() + 1000)),
        ),
        mock.patch(
            "requests.post",
            return_value=mock.MagicMock(
                json=mock.MagicMock(
                    return_value=_make_python_execution_response(result=2)
                )
            ),
        ),
        mock.patch.object(
            SessionsPythonREPLTool, "delete_session", autospec=True
        ) as mock_delete_session,
    ):
        agent.invoke({"messages": [{"role": "user", "content": "What is 1 + 1?"}]})

    mock_delete_session.assert_not_called()
    print("OK")


def test_session_deleted_for_each_tool_invocation() -> None:
    """When the agent makes multiple tool calls across turns,
    delete_session() is called once per tool invocation.
    """
    print("test_session_deleted_for_each_tool_invocation ... ", end="", flush=True)

    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    # Two separate turns that each trigger a tool call.
    fake_model = FakeListChatModel(
        responses=[
            _make_tool_call_ai_message(tool.name, {"__arg1": "2 + 2"}),
            _make_tool_call_ai_message(tool.name, {"__arg1": "3 + 3"}),
            AIMessage(content="Done."),
        ]
    )
    agent = create_agent(model=fake_model, tools=[tool])

    with (
        mock.patch(
            "azure.identity.DefaultAzureCredential.get_token",
            return_value=AccessToken("token_value", int(time.time() + 1000)),
        ),
        mock.patch(
            "requests.post",
            return_value=mock.MagicMock(
                json=mock.MagicMock(
                    return_value=_make_python_execution_response(result=4)
                )
            ),
        ),
        mock.patch.object(
            SessionsPythonREPLTool, "delete_session", autospec=True
        ) as mock_delete_session,
    ):
        agent.invoke(
            {"messages": [{"role": "user", "content": "Calculate two things."}]}
        )

    assert mock_delete_session.call_count == 2, (
        f"Expected 2 delete_session calls, got {mock_delete_session.call_count}"
    )
    print("OK")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_python_repl_session_deleted_after_agent_tool_call,
        test_bash_session_deleted_after_agent_tool_call,
        test_session_not_deleted_by_default,
        test_session_deleted_for_each_tool_invocation,
    ]

    for test_fn in tests:
        test_fn()

    print("\nAll session-deletion tests passed.")
