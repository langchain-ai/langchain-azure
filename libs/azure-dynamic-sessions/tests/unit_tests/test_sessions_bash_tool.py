import json
import re
import time
from unittest import mock
from urllib.parse import parse_qs, urlparse

from azure.core.credentials import AccessToken

from langchain_azure_dynamic_sessions import SessionsBashTool
from langchain_azure_dynamic_sessions.tools.sessions import (
    _sanitize_bash_input,
)

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


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_bash_execution_calls_api(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    tool = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    mock_post.return_value.json.return_value = _make_execution_response(
        stdout="hello world\n"
    )
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
    mock_post.assert_called_once_with(mock.ANY, headers=headers, json=body)

    called_headers = mock_post.call_args.kwargs["headers"]
    assert re.match(
        r"^langchain-azure-dynamic-sessions/\d+\.\d+\.\d+.* \(Language=Python\)",
        called_headers["User-Agent"],
    )

    called_api_url = mock_post.call_args.args[0]
    assert called_api_url.startswith(api_url)


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_uses_2025_api_version(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    tool = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    mock_post.return_value.json.return_value = _make_execution_response()
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
    mock_post.return_value.json.return_value = _make_execution_response()
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    tool.run("echo hello")

    call_url = mock_post.call_args.args[0]
    parsed_url = urlparse(call_url)
    call_identifier = parse_qs(parsed_url.query)["identifier"][0]
    assert call_identifier == "00000000-0000-0000-0000-000000000003"


def test_sanitizes_input() -> None:
    tool = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = _make_execution_response()
        tool.run("```bash\necho hello\n```")
        body = mock_post.call_args.kwargs["json"]
        assert body["shellCommand"] == "echo hello"


def test_does_not_sanitize_input() -> None:
    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT, sanitize_input=False
    )
    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = _make_execution_response()
        tool.run("```bash\necho hello\n```")
        body = mock_post.call_args.kwargs["json"]
        assert body["shellCommand"] == "```bash\necho hello\n```"


def test_uses_custom_access_token_provider() -> None:
    def custom_access_token_provider() -> str:
        return "custom_token"

    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        access_token_provider=custom_access_token_provider,
    )

    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = _make_execution_response()
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
    tool = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    mock_post.return_value.json.return_value = _make_execution_response()
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    tool.run("echo hello")

    body = mock_post.call_args.kwargs["json"]
    assert (
        "codeInputType" not in body
    ), "codeInputType is not supported by Shell session pools"
    assert (
        "executionType" not in body
    ), "executionType is not supported by Shell session pools"
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
    tool = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    mock_post.return_value.json.return_value = documented_response
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
    tool = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    mock_post.return_value.json.return_value = _make_execution_response(
        stderr="No such file or directory", exit_code=1
    )
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    result = tool.run("cat /nonexistent")

    assert json.loads(result)["exitCode"] == 1
