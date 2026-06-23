"""Sample: demonstrate and verify the delete_session_after_invocation feature.

This sample shows how ``delete_session_after_invocation`` works in
``SessionsPythonREPLTool`` and ``SessionsBashTool``.

When ``delete_session_after_invocation=True``, the tool calls
``delete_session()`` after every agent tool-call, preventing the session pool
from exhausting its slot limit.

The sample covers:

1. **SessionsPythonREPLTool** — session deleted after a Python code execution.
2. **SessionsBashTool** — session deleted after a bash command execution.
3. **No deletion by default** — when the flag is ``False`` (or omitted),
   ``delete_session()`` is never called.
4. **Multiple invocations** — each agent turn that triggers a tool call
   results in exactly one ``delete_session()`` call.

Usage (no API keys required — all HTTP calls are mocked)::

    python samples/dynamic-sessions/session_deletion_sample.py

All assertions are run in-process; an ``AssertionError`` is raised on failure.
"""

from __future__ import annotations

import time
from typing import Any, Dict
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


def _make_tool_call_ai_message(tool_name: str, tool_args: Dict[str, Any]) -> AIMessage:
    """Return an AIMessage with a tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": tool_name, "args": tool_args}],
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def demo_python_repl_session_deleted_after_invocation() -> None:
    """Show that SessionsPythonREPLTool deletes its session after each tool call
    when delete_session_after_invocation=True.
    """
    print("Scenario 1: Python REPL – session deleted after invocation ... ", end="", flush=True)

    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    fake_model = FakeListChatModel(
        responses=[
            _make_tool_call_ai_message(tool.name, {"python_code": "print(2 + 2)"}),
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
        ) as mock_delete,
    ):
        agent.invoke({"messages": [{"role": "user", "content": "What is 2 + 2?"}]})

    assert mock_delete.call_count == 1, (
        f"Expected delete_session to be called once, got {mock_delete.call_count}"
    )
    print("OK")


def demo_bash_session_deleted_after_invocation() -> None:
    """Show that SessionsBashTool deletes its session after each tool call
    when delete_session_after_invocation=True.
    """
    print("Scenario 2: Bash – session deleted after invocation ... ", end="", flush=True)

    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    fake_model = FakeListChatModel(
        responses=[
            _make_tool_call_ai_message(tool.name, {"bash_command": "echo hello"}),
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
        ) as mock_delete,
    ):
        agent.invoke({"messages": [{"role": "user", "content": "Run: echo hello"}]})

    assert mock_delete.call_count == 1, (
        f"Expected delete_session to be called once, got {mock_delete.call_count}"
    )
    print("OK")


def demo_session_not_deleted_by_default() -> None:
    """Show that when delete_session_after_invocation is omitted (default False),
    delete_session() is never called.
    """
    print("Scenario 3: No deletion by default ... ", end="", flush=True)

    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)

    fake_model = FakeListChatModel(
        responses=[
            _make_tool_call_ai_message(tool.name, {"python_code": "1 + 1"}),
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
        ) as mock_delete,
    ):
        agent.invoke({"messages": [{"role": "user", "content": "What is 1 + 1?"}]})

    assert mock_delete.call_count == 0, (
        f"Expected delete_session not to be called, got {mock_delete.call_count} call(s)"
    )
    print("OK")


def demo_session_deleted_for_each_invocation() -> None:
    """Show that when an agent makes multiple tool calls across turns,
    delete_session() is called once per invocation.
    """
    print("Scenario 4: Session deleted for each of multiple invocations ... ", end="", flush=True)

    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    fake_model = FakeListChatModel(
        responses=[
            _make_tool_call_ai_message(tool.name, {"python_code": "2 + 2"}),
            _make_tool_call_ai_message(tool.name, {"python_code": "3 + 3"}),
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
        ) as mock_delete,
    ):
        agent.invoke(
            {"messages": [{"role": "user", "content": "Calculate two things."}]}
        )

    assert mock_delete.call_count == 2, (
        f"Expected 2 delete_session calls, got {mock_delete.call_count}"
    )
    print("OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all scenarios."""
    demo_python_repl_session_deleted_after_invocation()
    demo_bash_session_deleted_after_invocation()
    demo_session_not_deleted_by_default()
    demo_session_deleted_for_each_invocation()
    print("\nAll scenarios passed.")


if __name__ == "__main__":
    main()
