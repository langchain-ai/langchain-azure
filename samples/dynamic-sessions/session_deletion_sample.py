"""Sample: demonstrate ``delete_session_after_invocation`` using ``create_agent``.

This sample wires Azure Dynamic Sessions tools into a LangChain agent built with
``create_agent`` and runs real end-to-end calls.

When ``delete_session_after_invocation=True``, the tool automatically calls
``delete_session()`` after each tool invocation and rotates ``tool.session_id``.

The sample covers four scenarios:

1. **Python REPL + deletion enabled** — session ID rotates after agent tool use.
2. **Bash + deletion enabled** — session ID rotates after agent tool use.
3. **Default behavior** — session ID remains stable across invocations.
4. **Multiple turns** — session ID rotates on every invocation.

Prerequisites::

    pip install langchain-azure-dynamic-sessions langchain-openai

Environment variables::

    AZURE_DYNAMIC_SESSIONS_POOL_MANAGEMENT_ENDPOINT (required)
        The management endpoint of your Dynamic Sessions pool.

    AZURE_OPENAI_ENDPOINT (required)
    AZURE_OPENAI_API_KEY (required)
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME (required)
    AZURE_OPENAI_API_VERSION (optional, defaults to 2024-12-01-preview)

Authentication for Dynamic Sessions uses ``DefaultAzureCredential``
(``az login`` / managed identity / service principal).

Usage::

    python samples/dynamic-sessions/session_deletion_sample.py
"""

from __future__ import annotations

import os
import time
from typing import Any

from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_azure_dynamic_sessions import SessionsBashTool, SessionsPythonREPLTool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POOL_MANAGEMENT_ENDPOINT = os.environ["AZURE_DYNAMIC_SESSIONS_POOL_MANAGEMENT_ENDPOINT"]


def _build_model() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0,
    )


def _agent_used_tool(result: dict[str, Any]) -> bool:
    messages = result.get("messages", [])
    return any(getattr(message, "type", "") == "tool" for message in messages)


def _invoke_agent_with_tool(tool: Any, user_prompt: str) -> dict[str, Any]:
    agent = create_agent(model=_build_model(), tools=[tool])
    return agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def demo_python_repl_session_deleted_after_invocation() -> None:
    """Python_REPL tool usage via create_agent rotates session_id when enabled."""
    print("Scenario 1: Python REPL via create_agent – session ID rotates after invocation")

    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    session_id_before = tool.session_id
    print(f"  session_id before: {session_id_before}")

    result = _invoke_agent_with_tool(
        tool,
        "Use the Python_REPL tool to compute 2 + 2. "
        "You must call the tool exactly once, then return only the numeric answer.",
    )

    session_id_after = tool.session_id
    print(f"  session_id after:  {session_id_after}")
    assert _agent_used_tool(result), "Agent did not call the tool"

    assert session_id_before != session_id_after, (
        "session_id should have rotated after invocation"
    )
    print("  OK – session_id rotated as expected\n")


def demo_bash_session_deleted_after_invocation() -> None:
    """Bash tool usage via create_agent rotates session_id when enabled."""
    print("Scenario 2: Bash via create_agent – session ID rotates after invocation")

    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    session_id_before = tool.session_id
    print(f"  session_id before: {session_id_before}")

    result = _invoke_agent_with_tool(
        tool,
        "Use the Bash tool to run: echo hello. "
        "You must call the tool exactly once, then summarize the output briefly.",
    )

    session_id_after = tool.session_id
    print(f"  session_id after:  {session_id_after}")
    assert _agent_used_tool(result), "Agent did not call the tool"

    assert session_id_before != session_id_after, (
        "session_id should have rotated after invocation"
    )
    print("  OK – session_id rotated as expected\n")


def demo_session_not_deleted_by_default() -> None:
    """With default delete_session_after_invocation=False, session_id is stable."""
    print("Scenario 3: No deletion by default (create_agent) – session ID is stable")

    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)

    session_id_before = tool.session_id
    print(f"  session_id: {session_id_before}")

    first = _invoke_agent_with_tool(
        tool,
        "Use the Python_REPL tool to compute 1 + 1. "
        "You must call the tool exactly once.",
    )
    second = _invoke_agent_with_tool(
        tool,
        "Use the Python_REPL tool to compute 2 + 2. "
        "You must call the tool exactly once.",
    )

    session_id_after = tool.session_id
    print(f"  session_id after two invocations: {session_id_after}")
    assert _agent_used_tool(first), "First agent call did not use the tool"
    assert _agent_used_tool(second), "Second agent call did not use the tool"

    assert session_id_before == session_id_after, (
        "session_id should remain stable when delete_session_after_invocation=False"
    )
    print("  OK – session_id unchanged as expected\n")

    # Clean up the persistent session explicitly.
    tool.delete_session()


def demo_session_deleted_for_each_invocation() -> None:
    """With deletion enabled, session_id rotates for each create_agent invocation."""
    print("Scenario 4: Session ID rotates on every create_agent invocation")

    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    ids = [tool.session_id]

    for expr in ("2 + 2", "3 + 3", "4 + 4"):
        result = _invoke_agent_with_tool(
            tool,
            f"Use the Python_REPL tool to compute {expr}. "
            "You must call the tool exactly once.",
        )
        assert _agent_used_tool(result), f"Agent did not call the tool for {expr}"
        ids.append(tool.session_id)

    print(f"  session_ids observed: {ids}")

    assert len(set(ids)) == len(ids), (
        "Every session_id should be unique – rotation must happen after each invoke"
    )
    print(f"  OK – {len(ids)} unique session IDs observed (one per invocation + initial)\n")

    # Allow background deletion threads to finish before the process exits.
    time.sleep(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all scenarios."""
    demo_python_repl_session_deleted_after_invocation()
    demo_bash_session_deleted_after_invocation()
    demo_session_not_deleted_by_default()
    demo_session_deleted_for_each_invocation()
    print("All scenarios passed.")


if __name__ == "__main__":
    main()
