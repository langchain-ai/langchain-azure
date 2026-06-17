"""Tests for the session_idle_timeout_seconds constructor parameter.

The Azure Dynamic Sessions data-plane API does not expose a per-session idle
timeout. ``SessionsPythonREPLTool`` and ``SessionsBashTool`` therefore implement
client-side auto-deletion: when ``session_idle_timeout_seconds`` is set, an
internal timer is (re)started after every successful API call and the session
is deleted via ``DELETE /session`` when the timer fires.
"""

import time
from unittest import mock

from azure.core.credentials import AccessToken

from langchain_azure_dynamic_sessions import (
    SessionsBashTool,
    SessionsPythonREPLTool,
)

POOL_MANAGEMENT_ENDPOINT = (
    "https://westus2.dynamicsessions.io/subscriptions/"
    "00000000-0000-0000-0000-000000000000/resourceGroups/"
    "sessions-rg/sessionPools/my-pool"
)


def _python_execute_response() -> dict:
    return {
        "id": "test-id",
        "identifier": "test-identifier",
        "status": "Succeeded",
        "properties": {
            "$id": "1",
            "status": "Success",
            "stdout": "",
            "stderr": "",
            "result": 42,
            "executionTimeInMilliseconds": 1,
        },
    }


def _bash_execute_response() -> dict:
    return {
        "identifier": "test-identifier",
        "status": "0",
        "result": {
            "stdout": "",
            "stderr": "",
            "executionTimeInMilliseconds": 1,
        },
    }


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_python_tool_idle_timeout_default_is_none() -> None:
    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    assert tool.session_idle_timeout_seconds is None


def test_bash_tool_idle_timeout_default_is_none() -> None:
    tool = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    assert tool.session_idle_timeout_seconds is None


def test_python_tool_no_timer_started_when_idle_timeout_none() -> None:
    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    with (
        mock.patch("requests.post") as mock_post,
        mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_token,
    ):
        mock_post.return_value.json.return_value = _python_execute_response()
        mock_token.return_value = AccessToken("token_value", int(time.time() + 1000))
        tool.execute("1 + 1")
    assert tool._idle_timer is None


def test_bash_tool_no_timer_started_when_idle_timeout_none() -> None:
    tool = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    with (
        mock.patch("requests.post") as mock_post,
        mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_token,
    ):
        mock_post.return_value.json.return_value = _bash_execute_response()
        mock_token.return_value = AccessToken("token_value", int(time.time() + 1000))
        tool.execute("echo hello")
    assert tool._idle_timer is None


# ---------------------------------------------------------------------------
# Timer lifecycle on activity
# ---------------------------------------------------------------------------


def test_python_tool_starts_timer_on_execute() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_idle_timeout_seconds=60.0,
    )
    try:
        with (
            mock.patch("requests.post") as mock_post,
            mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_token,
        ):
            mock_post.return_value.json.return_value = _python_execute_response()
            mock_token.return_value = AccessToken(
                "token_value", int(time.time() + 1000)
            )
            tool.execute("1 + 1")
        assert tool._idle_timer is not None
        assert tool._idle_timer.is_alive()
    finally:
        tool._cancel_idle_timer()


def test_bash_tool_starts_timer_on_execute() -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_idle_timeout_seconds=60.0,
    )
    try:
        with (
            mock.patch("requests.post") as mock_post,
            mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_token,
        ):
            mock_post.return_value.json.return_value = _bash_execute_response()
            mock_token.return_value = AccessToken(
                "token_value", int(time.time() + 1000)
            )
            tool.execute("echo hello")
        assert tool._idle_timer is not None
        assert tool._idle_timer.is_alive()
    finally:
        tool._cancel_idle_timer()


def test_python_tool_resets_timer_on_subsequent_activity() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_idle_timeout_seconds=60.0,
    )
    try:
        with (
            mock.patch("requests.post") as mock_post,
            mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_token,
        ):
            mock_post.return_value.json.return_value = _python_execute_response()
            mock_token.return_value = AccessToken(
                "token_value", int(time.time() + 1000)
            )
            tool.execute("1 + 1")
            first_timer = tool._idle_timer
            tool.execute("2 + 2")
            second_timer = tool._idle_timer
        assert first_timer is not None
        assert second_timer is not None
        assert first_timer is not second_timer
        # First timer must have been cancelled (is_alive returns False once
        # cancelled before it fired).
        assert not first_timer.is_alive()
        assert second_timer.is_alive()
    finally:
        tool._cancel_idle_timer()


# ---------------------------------------------------------------------------
# Auto-delete after idle period
# ---------------------------------------------------------------------------


def test_python_tool_auto_deletes_session_after_idle_period() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_idle_timeout_seconds=0.05,
    )
    with (
        mock.patch("requests.post") as mock_post,
        mock.patch("requests.delete") as mock_delete,
        mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_token,
    ):
        mock_post.return_value.json.return_value = _python_execute_response()
        mock_delete.return_value.raise_for_status.return_value = None
        mock_token.return_value = AccessToken("token_value", int(time.time() + 1000))

        tool.execute("1 + 1")

        # Wait for the idle timer to fire and the auto-delete to happen.
        deadline = time.time() + 2.0
        while time.time() < deadline and mock_delete.call_count == 0:
            time.sleep(0.02)

    assert mock_delete.call_count == 1
    delete_url = mock_delete.call_args.args[0]
    assert "/session?" in delete_url or "/session&" in delete_url


def test_bash_tool_auto_deletes_session_after_idle_period() -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_idle_timeout_seconds=0.05,
    )
    with (
        mock.patch("requests.post") as mock_post,
        mock.patch("requests.delete") as mock_delete,
        mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_token,
    ):
        mock_post.return_value.json.return_value = _bash_execute_response()
        mock_delete.return_value.raise_for_status.return_value = None
        mock_token.return_value = AccessToken("token_value", int(time.time() + 1000))

        tool.execute("echo hello")

        deadline = time.time() + 2.0
        while time.time() < deadline and mock_delete.call_count == 0:
            time.sleep(0.02)

    assert mock_delete.call_count == 1


# ---------------------------------------------------------------------------
# Timer cancellation on close / delete_session
# ---------------------------------------------------------------------------


def test_python_tool_close_cancels_idle_timer() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_idle_timeout_seconds=60.0,
    )
    with (
        mock.patch("requests.post") as mock_post,
        mock.patch("requests.delete") as mock_delete,
        mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_token,
    ):
        mock_post.return_value.json.return_value = _python_execute_response()
        mock_delete.return_value.raise_for_status.return_value = None
        mock_token.return_value = AccessToken("token_value", int(time.time() + 1000))

        tool.execute("1 + 1")
        assert tool._idle_timer is not None and tool._idle_timer.is_alive()
        tool.close()

    assert tool._idle_timer is None
    assert mock_delete.call_count == 1


def test_bash_tool_close_cancels_idle_timer() -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_idle_timeout_seconds=60.0,
    )
    with (
        mock.patch("requests.post") as mock_post,
        mock.patch("requests.delete") as mock_delete,
        mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_token,
    ):
        mock_post.return_value.json.return_value = _bash_execute_response()
        mock_delete.return_value.raise_for_status.return_value = None
        mock_token.return_value = AccessToken("token_value", int(time.time() + 1000))

        tool.execute("echo hello")
        assert tool._idle_timer is not None and tool._idle_timer.is_alive()
        tool.close()

    assert tool._idle_timer is None
    assert mock_delete.call_count == 1


def test_python_tool_context_manager_cancels_idle_timer() -> None:
    with (
        mock.patch("requests.post") as mock_post,
        mock.patch("requests.delete") as mock_delete,
        mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_token,
    ):
        mock_post.return_value.json.return_value = _python_execute_response()
        mock_delete.return_value.raise_for_status.return_value = None
        mock_token.return_value = AccessToken("token_value", int(time.time() + 1000))

        with SessionsPythonREPLTool(
            pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
            session_idle_timeout_seconds=60.0,
        ) as tool:
            tool.execute("1 + 1")
            assert tool._idle_timer is not None and tool._idle_timer.is_alive()

    assert tool._idle_timer is None
    assert mock_delete.call_count == 1
