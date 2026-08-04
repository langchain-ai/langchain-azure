import json
import re
import threading
import time
from unittest import mock
from urllib.parse import parse_qs, urlparse

import pytest
from azure.core.credentials import AccessToken

from langchain_azure_container_apps.dynamic_sessions import SessionsBashTool
from langchain_azure_container_apps.dynamic_sessions._common import (
    REQUEST_TIMEOUT,
    access_token_provider_factory,
)
from langchain_azure_container_apps.dynamic_sessions.tools.sessions import (
    _sanitize_bash_input,
)
from tests.unit_tests.dynamic_sessions._helpers import stub_response

# Captured before any test patches `threading.Thread`, so it can be
# used as a spec.
_REAL_THREAD = threading.Thread

POOL_MANAGEMENT_ENDPOINT = "https://westus2.dynamicsessions.io/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/sessions-rg/sessionPools/my-pool"


def _make_execution_response(
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
) -> dict:
    # Response structure as documented at:
    # https://learn.microsoft.com/en-us/azure/container-apps/sessions-tutorial-shell
    return {
        "identifier": "test-identifier",
        "status": str(exit_code),
        "result": {
            "stdout": stdout,
            "stderr": stderr,
            "executionTimeInMilliseconds": 1,
        },
    }


def test_sanitize_bash_input_strips_backticks() -> None:
    assert _sanitize_bash_input("```bash\necho hello\n```") == "echo hello"


def test_sanitize_bash_input_strips_sh_prefix() -> None:
    assert _sanitize_bash_input("```sh\necho hello\n```") == "echo hello"


def test_sanitize_bash_input_strips_whitespace() -> None:
    assert _sanitize_bash_input("  echo hello  ") == "echo hello"


def test_sanitize_bash_input_no_prefix() -> None:
    assert _sanitize_bash_input("echo hello") == "echo hello"


def test_sanitize_bash_input_keeps_words_starting_with_sh() -> None:
    # The interpreter word is only stripped at a word boundary: commands that
    # merely start with "sh" used to be mangled ("shopt" -> "opt").
    assert _sanitize_bash_input("shopt -s nullglob") == "shopt -s nullglob"
    assert _sanitize_bash_input("sha256sum f.txt") == "sha256sum f.txt"
    assert _sanitize_bash_input("shift 2") == "shift 2"


def test_sanitize_bash_input_strips_leading_bash_word() -> None:
    assert _sanitize_bash_input("bash echo hi") == "echo hi"


def test_sanitize_bash_input_strips_leading_sh_word() -> None:
    # Documented pre-existing behavior: a standalone interpreter word followed
    # by whitespace IS stripped, even when the rest is its arguments.
    assert _sanitize_bash_input("sh -c 'x'") == "-c 'x'"


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_bash_execution_calls_api(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    stub_response(mock_post, _make_execution_response(stdout="hello world\n"))
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    result = tool.run("echo hello world")

    assert json.loads(result) == {
        "stdout": "hello world\n",
        "stderr": "",
        "exitCode": 0,
    }

    api_url = f"{POOL_MANAGEMENT_ENDPOINT}/executions"
    headers = {
        "Authorization": "Bearer token_value",
        "Content-Type": "application/json",
        "User-Agent": mock.ANY,
    }
    body = {
        "shellCommand": "echo hello world",
    }
    mock_post.assert_called_once_with(
        mock.ANY, headers=headers, json=body, timeout=REQUEST_TIMEOUT
    )

    called_headers = mock_post.call_args.kwargs["headers"]
    assert re.match(
        r"^langchain-azure-container-apps/\d+\.\d+\.\d+.* \(Language=Python\)",
        called_headers["User-Agent"],
    )

    called_api_url = mock_post.call_args.args[0]
    assert called_api_url.startswith(api_url)


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_uses_2025_api_version(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    stub_response(mock_post, _make_execution_response())
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    tool.run("echo hello")

    called_api_url = mock_post.call_args.args[0]
    parsed_url = urlparse(called_api_url)
    api_version = parse_qs(parsed_url.query)["api-version"][0]
    assert api_version == "2025-02-02-preview"


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_uses_specified_session_id(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_id="00000000-0000-0000-0000-000000000003",
    )
    stub_response(mock_post, _make_execution_response())
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    tool.run("echo hello")

    call_url = mock_post.call_args.args[0]
    parsed_url = urlparse(call_url)
    call_identifier = parse_qs(parsed_url.query)["identifier"][0]
    assert call_identifier == "00000000-0000-0000-0000-000000000003"


@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_sanitizes_input(mock_get_token: mock.MagicMock) -> None:
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    with mock.patch("requests.post") as mock_post:
        stub_response(mock_post, _make_execution_response())
        tool.run("```bash\necho hello\n```")
        body = mock_post.call_args.kwargs["json"]
        assert body["shellCommand"] == "echo hello"


def test_does_not_sanitize_input() -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT, sanitize_input=False
    )
    with mock.patch("requests.post") as mock_post:
        stub_response(mock_post, _make_execution_response())
        tool.run("```bash\necho hello\n```")
        body = mock_post.call_args.kwargs["json"]
        assert body["shellCommand"] == "```bash\necho hello\n```"


def test_each_instance_gets_unique_session_id() -> None:
    tool1 = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    tool2 = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    assert tool1.session_id != tool2.session_id


def test_uses_custom_access_token_provider() -> None:
    def custom_access_token_provider() -> str:
        return "custom_token"

    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        access_token_provider=custom_access_token_provider,
    )

    with mock.patch("requests.post") as mock_post:
        stub_response(mock_post, _make_execution_response())
        tool.run("echo hello")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer custom_token"


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_request_body_does_not_contain_unsupported_fields(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    """Shell session pools reject codeInputType and executionType fields.

    The Shell session pool API only supports 'shellCommand' (and optionally
    'timeoutInSeconds') in the request body. Fields 'codeInputType' and
    'executionType' are only valid for Python/code-typed session pools and
    result in a 400 Bad Request when sent to a Shell session pool.

    Reference:
    https://learn.microsoft.com/en-us/azure/container-apps/sessions-tutorial-shell
    """
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    stub_response(mock_post, _make_execution_response())
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    tool.run("echo hello")

    body = mock_post.call_args.kwargs["json"]
    assert "codeInputType" not in body, (
        "codeInputType is not supported by Shell session pools"
    )
    assert "executionType" not in body, (
        "executionType is not supported by Shell session pools"
    )
    assert set(body.keys()) == {"shellCommand"}


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_response_parsing_matches_documented_api_response(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    """Tool correctly parses the documented Shell session pool API response.

    The Shell session pool API returns exit code in the top-level 'status'
    field as a string (e.g. '0'), not in an 'exitCode' field.

    Example documented response:
    {
        "identifier": "...",
        "status": "0",
        "result": {
            "stdout": "Hello world!\\n",
            "stderr": "",
            "executionTimeInMilliseconds": 1
        }
    }

    Reference:
    https://learn.microsoft.com/en-us/azure/container-apps/sessions-tutorial-shell
    """
    # Use the exact response structure from the MS Learn documentation / issue report
    documented_response = {
        "identifier": "a32ba57a-db4d-4a56-b080-7818199dd105",
        "status": "0",
        "result": {
            "stdout": "Hello world!\n",
            "stderr": "",
            "executionTimeInMilliseconds": 1,
        },
    }
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    stub_response(mock_post, documented_response)
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    result = tool.run("echo Hello world!")

    assert json.loads(result) == {
        "stdout": "Hello world!\n",
        "stderr": "",
        "exitCode": 0,
    }


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_nonzero_exit_code_is_parsed_from_status(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    """Non-zero exit codes are parsed correctly from the 'status' field."""
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    stub_response(
        mock_post,
        _make_execution_response(stderr="No such file or directory", exit_code=1),
    )
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    result = tool.run("cat /nonexistent")

    assert json.loads(result)["exitCode"] == 1


@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_delete_session_resets_id_and_runs_async(
    mock_get_token: mock.MagicMock,
) -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_id="00000000-0000-0000-0000-000000000003",
    )
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    with mock.patch(
        "langchain_azure_container_apps.dynamic_sessions._base.threading.Thread"
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
    tool = SessionsBashTool(
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
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )

    with mock.patch.object(
        SessionsBashTool, "delete_session", autospec=True
    ) as mock_delete_session:
        tool.close()

    mock_delete_session.assert_called_once_with(tool)


def test_context_manager_closes_session() -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )

    with mock.patch.object(SessionsBashTool, "close", autospec=True) as mock_close:
        with tool as entered_tool:
            assert entered_tool is tool

    mock_close.assert_called_once_with(tool)


def test_does_not_delete_session_after_invocation_by_default() -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )

    with (
        mock.patch.object(SessionsBashTool, "execute", autospec=True) as mock_execute,
        mock.patch.object(
            SessionsBashTool, "delete_session", autospec=True
        ) as mock_delete_session,
    ):
        mock_execute.return_value = _make_execution_response()

        tool.run("echo hello")

    mock_delete_session.assert_not_called()


def test_deletes_session_after_invocation_when_configured() -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        delete_session_after_invocation=True,
    )

    with (
        mock.patch.object(SessionsBashTool, "execute", autospec=True) as mock_execute,
        mock.patch.object(
            SessionsBashTool, "delete_session", autospec=True
        ) as mock_delete_session,
    ):
        mock_execute.return_value = _make_execution_response()

        tool.run("echo hello")

    mock_delete_session.assert_called_once_with(tool)


def test_build_url_requires_an_endpoint() -> None:
    tool = SessionsBashTool(pool_management_endpoint="")
    with pytest.raises(ValueError, match="pool_management_endpoint is not set"):
        tool._build_url("executions")


def test_endpoint_with_a_query_string_is_rejected() -> None:
    tool = SessionsBashTool(pool_management_endpoint=f"{POOL_MANAGEMENT_ENDPOINT}?x=1")
    with pytest.raises(ValueError, match="must not contain a query string"):
        tool._build_url("executions")


@mock.patch("requests.delete")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_delete_worker_calls_the_api(
    mock_get_token: mock.MagicMock, mock_delete: mock.MagicMock
) -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    with mock.patch(
        "langchain_azure_container_apps.dynamic_sessions._base.threading.Thread"
    ) as mock_thread:
        tool.delete_session()

    mock_thread.call_args.kwargs["target"]()

    assert mock_delete.call_count == 1


def test_delete_worker_swallows_failures() -> None:
    # Deletion is fire-and-forget on a daemon thread; an exception escaping the
    # worker would surface as an unraisable thread exception, not a tool error.
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )

    with mock.patch(
        "langchain_azure_container_apps.dynamic_sessions._base.threading.Thread"
    ) as mock_thread:
        tool.delete_session()

    with mock.patch.object(
        SessionsBashTool, "_delete_session_sync", side_effect=RuntimeError("boom")
    ):
        mock_thread.call_args.kwargs["target"]()


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_unparseable_status_yields_a_null_exit_code(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    stub_response(
        mock_post,
        {
            "identifier": "test-identifier",
            "status": "Succeeded",
            "result": {"stdout": "hi\n", "stderr": ""},
        },
    )
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    result = tool.run("echo hi")

    assert json.loads(result)["exitCode"] is None


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_missing_status_yields_a_null_exit_code(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        # A fresh provider closure, not the shared class-level default: its
        # token cache survives across tests, and a real token cached by an
        # earlier live test would shadow the patched `get_token` here.
        access_token_provider=access_token_provider_factory(),
    )
    stub_response(mock_post, {"result": {"stdout": "", "stderr": ""}})
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    assert json.loads(tool.run("echo hi"))["exitCode"] is None
