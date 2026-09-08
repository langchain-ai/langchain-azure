import json
import re
import threading
import time
from unittest import mock
from urllib.parse import parse_qs, urlparse

import pytest
from azure.core.credentials import AccessToken
from pydantic import ValidationError

from langchain_azure_compute.dynamic_sessions import SessionsPythonREPLTool
from langchain_azure_compute.dynamic_sessions._common import (
    REQUEST_TIMEOUT,
    access_token_provider_factory,
)
from langchain_azure_compute.dynamic_sessions.tools.sessions import (
    _sanitize_input,
)
from tests.unit_tests.dynamic_sessions._helpers import stub_response

# Captured before any test patches `threading.Thread`, so it can be
# used as a spec.
_REAL_THREAD = threading.Thread

POOL_MANAGEMENT_ENDPOINT = "https://westus2.dynamicsessions.io/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/sessions-rg/sessionPools/my-pool"


def test_sanitize_input_strips_backticks() -> None:
    assert _sanitize_input("```python\nprint(1)\n```") == "print(1)"


def test_sanitize_input_keeps_words_starting_with_python() -> None:
    # The interpreter word is only stripped at a word boundary: identifiers
    # that merely start with "python" used to be mangled
    # ("python_var" -> "_var").
    assert _sanitize_input("python_var = 1") == "python_var = 1"
    assert _sanitize_input("pythonic()") == "pythonic()"


def test_default_access_token_provider_returns_token() -> None:
    access_token_provider = access_token_provider_factory()
    with mock.patch(
        "azure.identity.DefaultAzureCredential.get_token"
    ) as mock_get_token:
        mock_get_token.return_value = AccessToken("token_value", 0)
        access_token = access_token_provider()
        assert access_token == "token_value"


def test_default_access_token_provider_returns_cached_token() -> None:
    access_token_provider = access_token_provider_factory()
    with mock.patch(
        "azure.identity.DefaultAzureCredential.get_token"
    ) as mock_get_token:
        mock_get_token.return_value = AccessToken(
            "token_value", int(time.time() + 1000)
        )
        access_token = access_token_provider()
        assert access_token == "token_value"
        assert mock_get_token.call_count == 1

        mock_get_token.return_value = AccessToken(
            "new_token_value", int(time.time() + 1000)
        )
        access_token = access_token_provider()
        assert access_token == "token_value"
        assert mock_get_token.call_count == 1


def test_default_access_token_provider_refreshes_expiring_token() -> None:
    access_token_provider = access_token_provider_factory()
    with mock.patch(
        "azure.identity.DefaultAzureCredential.get_token"
    ) as mock_get_token:
        mock_get_token.return_value = AccessToken("token_value", int(time.time() - 1))
        access_token = access_token_provider()
        assert access_token == "token_value"
        assert mock_get_token.call_count == 1

        mock_get_token.return_value = AccessToken(
            "new_token_value", int(time.time() + 1000)
        )
        access_token = access_token_provider()
        assert access_token == "new_token_value"
        assert mock_get_token.call_count == 2


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_code_execution_calls_api(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    stub_response(
        mock_post,
        {
            "$id": "1",
            "properties": {
                "$id": "2",
                "status": "Success",
                "stdout": "hello world\n",
                "stderr": "",
                "result": "",
                "executionTimeInMilliseconds": 33,
            },
        },
    )
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    result = tool.run("print('hello world')")

    assert json.loads(result) == {
        "result": "",
        "stdout": "hello world\n",
        "stderr": "",
    }

    api_url = f"{POOL_MANAGEMENT_ENDPOINT}/code/execute"
    headers = {
        "Authorization": "Bearer token_value",
        "Content-Type": "application/json",
        "User-Agent": mock.ANY,
    }
    body = {
        "properties": {
            "codeInputType": "inline",
            "executionType": "synchronous",
            "code": "print('hello world')",
        }
    }
    mock_post.assert_called_once_with(
        mock.ANY, headers=headers, json=body, timeout=REQUEST_TIMEOUT
    )

    called_headers = mock_post.call_args.kwargs["headers"]
    assert re.match(
        r"^langchain-azure-compute/\d+\.\d+\.\d+.* \(Language=Python\)",
        called_headers["User-Agent"],
    )

    called_api_url = mock_post.call_args.args[0]
    assert called_api_url.startswith(api_url)


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_request_timeout_is_configurable(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    """Executions are synchronous server-side, so code running longer than the
    120 s default needs a per-instance escape hatch."""
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        access_token_provider=access_token_provider_factory(),
        request_timeout=600,
    )
    stub_response(
        mock_post,
        {"properties": {"status": "Success", "stdout": "", "stderr": "", "result": ""}},
    )
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))
    tool.run("import time; time.sleep(300)")
    assert mock_post.call_args.kwargs["timeout"] == 600


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_request_timeout_is_rejected_at_construction(bad: int) -> None:
    """Requests raises on a zero or negative timeout at the first call, which
    would surface far from the misconfiguration that caused it."""
    with pytest.raises(ValidationError):
        SessionsPythonREPLTool(
            pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
            access_token_provider=access_token_provider_factory(),
            request_timeout=bad,
        )


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_uses_specified_session_id(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_id="00000000-0000-0000-0000-000000000003",
    )
    stub_response(
        mock_post,
        {
            "$id": "1",
            "properties": {
                "$id": "2",
                "status": "Success",
                "stdout": "",
                "stderr": "",
                "result": "2",
                "executionTimeInMilliseconds": 33,
            },
        },
    )
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))
    tool.run("1 + 1")
    call_url = mock_post.call_args.args[0]
    parsed_url = urlparse(call_url)
    call_identifier = parse_qs(parsed_url.query)["identifier"][0]
    assert call_identifier == "00000000-0000-0000-0000-000000000003"


@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_sanitizes_input(mock_get_token: mock.MagicMock) -> None:
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    with mock.patch("requests.post") as mock_post:
        stub_response(
            mock_post,
            {
                "$id": "1",
                "properties": {
                    "$id": "2",
                    "status": "Success",
                    "stdout": "",
                    "stderr": "",
                    "result": "",
                    "executionTimeInMilliseconds": 33,
                },
            },
        )
        tool.run("```python\nprint('hello world')\n```")
        body = mock_post.call_args.kwargs["json"]
        assert body["properties"]["code"] == "print('hello world')"


def test_does_not_sanitize_input() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT, sanitize_input=False
    )
    with mock.patch("requests.post") as mock_post:
        stub_response(
            mock_post,
            {
                "$id": "1",
                "properties": {
                    "$id": "2",
                    "status": "Success",
                    "stdout": "",
                    "stderr": "",
                    "result": "",
                    "executionTimeInMilliseconds": 33,
                },
            },
        )
        tool.run("```python\nprint('hello world')\n```")
        body = mock_post.call_args.kwargs["json"]
        assert body["properties"]["code"] == "```python\nprint('hello world')\n```"


def test_each_instance_gets_unique_session_id() -> None:
    tool1 = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    tool2 = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    assert tool1.session_id != tool2.session_id


def test_uses_custom_access_token_provider() -> None:
    def custom_access_token_provider() -> str:
        return "custom_token"

    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        access_token_provider=custom_access_token_provider,
    )

    with mock.patch("requests.post") as mock_post:
        stub_response(
            mock_post,
            {
                "$id": "1",
                "properties": {
                    "$id": "2",
                    "status": "Success",
                    "stdout": "",
                    "stderr": "",
                    "result": "",
                    "executionTimeInMilliseconds": 33,
                },
            },
        )
        tool.run("print('hello world')")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer custom_token"


@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_delete_session_resets_id_and_runs_async(
    mock_get_token: mock.MagicMock,
) -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_id="00000000-0000-0000-0000-000000000003",
    )
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    with mock.patch(
        "langchain_azure_compute.dynamic_sessions._base.threading.Thread"
    ) as mock_thread:
        mock_thread_instance = mock.Mock(spec=_REAL_THREAD)
        mock_thread.return_value = mock_thread_instance

        original_session_id = tool.session_id
        tool.delete_session()

    assert tool.session_id != original_session_id
    mock_thread.assert_called_once()
    assert mock_thread.call_args.kwargs["daemon"] is True
    mock_thread_instance.start.assert_called_once()


@mock.patch("requests.delete")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_delete_session_sync_calls_api(
    mock_get_token: mock.MagicMock, mock_delete: mock.MagicMock
) -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    tool._delete_session_sync("00000000-0000-0000-0000-000000000003")

    mock_delete.assert_called_once_with(
        mock.ANY,
        headers={
            "Authorization": mock.ANY,
            "User-Agent": mock.ANY,
        },
        timeout=REQUEST_TIMEOUT,
    )
    called_headers = mock_delete.call_args.kwargs["headers"]
    assert called_headers["Authorization"].endswith("token_value")
    called_api_url = mock_delete.call_args.args[0]
    assert called_api_url.startswith(f"{POOL_MANAGEMENT_ENDPOINT}/session")
    parsed_url = urlparse(called_api_url)
    parsed_qs = parse_qs(parsed_url.query)
    assert parsed_qs["identifier"][0] == "00000000-0000-0000-0000-000000000003"
    assert parsed_qs["api-version"][0] == "2025-10-02-preview"


def test_close_calls_delete_session() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )

    with mock.patch.object(
        SessionsPythonREPLTool, "delete_session", autospec=True
    ) as mock_delete_session:
        tool.close()

    mock_delete_session.assert_called_once_with(tool)


def test_context_manager_closes_session() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )

    with mock.patch.object(
        SessionsPythonREPLTool, "close", autospec=True
    ) as mock_close:
        with tool as entered_tool:
            assert entered_tool is tool

    mock_close.assert_called_once_with(tool)


def test_does_not_delete_session_after_invocation_by_default() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )

    with (
        mock.patch.object(
            SessionsPythonREPLTool, "execute", autospec=True
        ) as mock_execute,
        mock.patch.object(
            SessionsPythonREPLTool, "delete_session", autospec=True
        ) as mock_delete_session,
    ):
        mock_execute.return_value = {
            "result": "2",
            "stdout": "",
            "stderr": "",
        }

        tool.run("1 + 1")

    mock_delete_session.assert_not_called()


def test_deletes_session_after_invocation_when_configured() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    with (
        mock.patch.object(
            SessionsPythonREPLTool, "execute", autospec=True
        ) as mock_execute,
        mock.patch.object(
            SessionsPythonREPLTool, "delete_session", autospec=True
        ) as mock_delete_session,
    ):
        mock_execute.return_value = {
            "result": "2",
            "stdout": "",
            "stderr": "",
        }

        tool.run("1 + 1")

    mock_delete_session.assert_called_once_with(tool)


def test_build_url_requires_an_endpoint() -> None:
    tool = SessionsPythonREPLTool(pool_management_endpoint="")
    with pytest.raises(ValueError, match="pool_management_endpoint is not set"):
        tool._build_url("code/execute")


@mock.patch("requests.delete")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_delete_worker_calls_the_api(
    mock_get_token: mock.MagicMock, mock_delete: mock.MagicMock
) -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    with mock.patch(
        "langchain_azure_compute.dynamic_sessions._base.threading.Thread"
    ) as mock_thread:
        tool.delete_session()

    mock_thread.call_args.kwargs["target"]()

    assert mock_delete.call_count == 1


def test_delete_worker_swallows_failures() -> None:
    # Deletion is fire-and-forget on a daemon thread; an exception escaping the
    # worker would surface as an unraisable thread exception, not a tool error.
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )

    with mock.patch(
        "langchain_azure_compute.dynamic_sessions._base.threading.Thread"
    ) as mock_thread:
        tool.delete_session()

    with mock.patch.object(
        SessionsPythonREPLTool, "_delete_session_sync", side_effect=RuntimeError("boom")
    ):
        mock_thread.call_args.kwargs["target"]()


def test_image_base64_payload_is_stripped_from_content() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    response = {
        "result": {"type": "image", "format": "png", "base64_data": "iVBORw0KGgo="},
        "stdout": "",
        "stderr": "",
    }

    with mock.patch.object(SessionsPythonREPLTool, "execute", return_value=response):
        content, artifact = tool._run("plot()")

    assert "base64_data" not in json.loads(content)["result"]
    # The artifact keeps the full payload; only the model-facing content is trimmed.
    assert artifact["result"]["base64_data"] == "iVBORw0KGgo="


def test_image_base64_payload_is_kept_when_requested() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    response = {
        "result": {"type": "image", "base64_data": "iVBORw0KGgo="},
        "stdout": "",
        "stderr": "",
    }

    with mock.patch.object(SessionsPythonREPLTool, "execute", return_value=response):
        content, _ = tool._run("plot()", remove_image_base64=False)

    assert json.loads(content)["result"]["base64_data"] == "iVBORw0KGgo="


def test_non_image_dict_result_is_left_alone() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    response = {"result": {"type": "table", "rows": 3}, "stdout": "", "stderr": ""}

    with mock.patch.object(SessionsPythonREPLTool, "execute", return_value=response):
        content, _ = tool._run("df")

    assert json.loads(content)["result"] == {"type": "table", "rows": 3}
