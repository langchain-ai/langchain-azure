"""Sample: demonstrate the delete_session_after_invocation feature.

This sample shows how ``delete_session_after_invocation`` works in
``SessionsPythonREPLTool`` and ``SessionsBashTool`` by invoking the tools
directly against a real Azure Dynamic Sessions pool.

When ``delete_session_after_invocation=True``, the tool automatically calls
``delete_session()`` after each invocation, which:

- Sends a DELETE request to remove the session from the pool (async, in a
  background thread).
- Rotates ``tool.session_id`` immediately so the next invocation starts with
  a fresh session.

The sample covers four scenarios:

1. **SessionsPythonREPLTool with deletion** — session ID rotates after invoke.
2. **SessionsBashTool with deletion** — session ID rotates after invoke.
3. **No deletion by default** — session ID is stable across invocations.
4. **Multiple invocations** — session ID changes after every invoke.

Prerequisites::

    pip install langchain-azure-dynamic-sessions

Environment variables::

    AZURE_DYNAMIC_SESSIONS_POOL_MANAGEMENT_ENDPOINT  (required)
        The management endpoint of your Azure Dynamic Sessions pool.
        Example:
          https://westus2.dynamicsessions.io/subscriptions/<sub>/resourceGroups/<rg>/sessionPools/<pool>

    Authentication is handled by ``DefaultAzureCredential`` — run
    ``az login`` (or set up a managed identity / service principal) before
    executing the sample.

Usage::

    python samples/dynamic-sessions/session_deletion_sample.py
"""

from __future__ import annotations

import os
import time

from langchain_azure_dynamic_sessions import SessionsBashTool, SessionsPythonREPLTool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POOL_MANAGEMENT_ENDPOINT = os.environ["AZURE_DYNAMIC_SESSIONS_POOL_MANAGEMENT_ENDPOINT"]

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def demo_python_repl_session_deleted_after_invocation() -> None:
    """SessionsPythonREPLTool rotates its session_id after each invocation
    when delete_session_after_invocation=True.
    """
    print("Scenario 1: Python REPL – session ID rotates after invocation")

    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    session_id_before = tool.session_id
    print(f"  session_id before: {session_id_before}")

    result, _ = tool.invoke("2 + 2")
    print(f"  result: {result.strip()}")

    session_id_after = tool.session_id
    print(f"  session_id after:  {session_id_after}")

    assert session_id_before != session_id_after, (
        "session_id should have rotated after invocation"
    )
    print("  OK – session_id rotated as expected\n")


def demo_bash_session_deleted_after_invocation() -> None:
    """SessionsBashTool rotates its session_id after each invocation
    when delete_session_after_invocation=True.
    """
    print("Scenario 2: Bash – session ID rotates after invocation")

    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    session_id_before = tool.session_id
    print(f"  session_id before: {session_id_before}")

    result, _ = tool.invoke("echo hello")
    print(f"  result: {result.strip()}")

    session_id_after = tool.session_id
    print(f"  session_id after:  {session_id_after}")

    assert session_id_before != session_id_after, (
        "session_id should have rotated after invocation"
    )
    print("  OK – session_id rotated as expected\n")


def demo_session_not_deleted_by_default() -> None:
    """When delete_session_after_invocation is omitted (default False),
    the session_id stays the same across invocations.
    """
    print("Scenario 3: No deletion by default – session ID is stable")

    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)

    session_id_before = tool.session_id
    print(f"  session_id: {session_id_before}")

    tool.invoke("1 + 1")
    tool.invoke("2 + 2")

    session_id_after = tool.session_id
    print(f"  session_id after two invocations: {session_id_after}")

    assert session_id_before == session_id_after, (
        "session_id should remain stable when delete_session_after_invocation=False"
    )
    print("  OK – session_id unchanged as expected\n")

    # Clean up the persistent session explicitly.
    tool.delete_session()


def demo_session_deleted_for_each_invocation() -> None:
    """Each invocation rotates the session_id when
    delete_session_after_invocation=True.
    """
    print("Scenario 4: Session ID rotates on every invocation")

    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    ids = [tool.session_id]

    for expr in ("2 + 2", "3 + 3", "4 + 4"):
        tool.invoke(expr)
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
