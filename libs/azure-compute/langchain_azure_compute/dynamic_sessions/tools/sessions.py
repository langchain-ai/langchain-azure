"""The Azure Dynamic Sessions tools.

Provides `SessionsPythonREPLTool` and `SessionsBashTool` for executing code in
Azure Container Apps dynamic sessions.
"""

import json
import logging
import re
import urllib.parse
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import Any, BinaryIO, ClassVar, List, Optional, Tuple

import requests

from langchain_azure_compute.dynamic_sessions._base import _BaseSessionsTool
from langchain_azure_compute.dynamic_sessions._common import (
    USER_AGENT,
)

__all__ = [
    "RemoteFileMetadata",
    "SessionsBashTool",
    "SessionsPythonREPLTool",
    # Historical re-export: the old package exposed USER_AGENT from its tool
    # module, and code migrating from it may import it from here.
    "USER_AGENT",
]

logger = logging.getLogger(__name__)


def _sanitize_input(query: str) -> str:
    """Strip code fences and a leading ``python`` word from REPL input.

    The interpreter word is only stripped when it stands alone (followed by
    whitespace, a backtick, or the end of input): without that boundary,
    ``python_var = 1`` used to be mangled into ``_var = 1``.

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """
    query = re.sub(r"^(\s|`)*(?:(?i:python)(?=\s|`|$))?\s*", "", query)
    query = re.sub(r"(\s|`)*$", "", query)
    return query


def _sanitize_bash_input(query: str) -> str:
    """Strip code fences and a leading ``bash``/``sh`` word from shell input.

    The interpreter word is only stripped when it stands alone (followed by
    whitespace, a backtick, or the end of input): without that boundary,
    ``shopt -s nullglob`` used to be mangled into ``opt -s nullglob`` and
    ``sha256sum f`` into ``a256sum f``.

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """
    query = re.sub(r"^(\s|`)*(?:(?i:bash|sh)(?=\s|`|$))?\s*", "", query)
    query = re.sub(r"(\s|`)*$", "", query)
    return query


@dataclass
class RemoteFileMetadata:
    """Metadata for a file in the session."""

    filename: str
    """The filename relative to `/mnt/data`."""

    size_in_bytes: int
    """The size of the file in bytes."""

    @property
    def full_path(self) -> str:
        """Get the full path of the file."""
        return f"/mnt/data/{self.filename}"

    @staticmethod
    def from_dict(data: dict) -> "RemoteFileMetadata":
        """Create a RemoteFileMetadata object from a dictionary."""
        properties = data.get("properties", {})
        return RemoteFileMetadata(
            filename=properties.get("filename"),
            size_in_bytes=properties.get("size"),
        )


class SessionsPythonREPLTool(_BaseSessionsTool):
    r"""Azure Dynamic Sessions tool.

    **Setup:**

    ```bash
    pip install -U "langchain-azure-compute[dynamic-sessions]"
    ```

    ```python
    import getpass

    POOL_MANAGEMENT_ENDPOINT = getpass.getpass("Enter the management endpoint of the session pool: ")
    ```

    **Instantiation:**

    ```python

    from langchain_azure_compute.dynamic_sessions import SessionsPythonREPLTool

    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT
    )
    ```

    **Invocation with args:**

    ```python
    tool.invoke("6 * 7")
    ```

    ```output
    '{\\n  "result": 42,\\n  "stdout": "",\\n  "stderr": ""\\n}'
    ```

    **Invocation with ToolCall:**

    ```python
    tool.invoke({"args": {"input":"6 * 7"}, "id": "1", "name": tool.name, "type": "tool_call"})
    ```

    ```output
    '{\\n  "result": 42,\\n  "stdout": "",\\n  "stderr": ""\\n}'
    ```
    """  # noqa: E501

    name: str = "Python_REPL"
    description: str = (
        "A Python shell. Use this to execute python commands "
        "when you need to perform calculations or computations. "
        "Input should be a valid python command. "
        "Returns a JSON object with the result, stdout, and stderr. "
    )

    _API_VERSION: ClassVar[str] = "2024-02-02-preview"

    def execute(self, python_code: str) -> Any:
        """Execute Python code in the session."""
        if self.sanitize_input:
            python_code = _sanitize_input(python_code)

        api_url = self._build_url("code/execute")
        headers = self._auth_headers(json_body=True)
        body = {
            "properties": {
                "codeInputType": "inline",
                "executionType": "synchronous",
                "code": python_code,
            }
        }

        response = requests.post(
            api_url, headers=headers, json=body, timeout=self.request_timeout
        )
        response.raise_for_status()
        response_json = response.json()
        properties = response_json.get("properties", {})
        return properties

    def _run(
        self, python_code: str, remove_image_base64: bool = True, **kwargs: Any
    ) -> Tuple[str, dict]:
        try:
            response = self.execute(python_code)

            # if the result is an image, remove the base64 data
            result = deepcopy(response.get("result"))
            if isinstance(result, dict) and remove_image_base64:
                if result.get("type") == "image" and "base64_data" in result:
                    result.pop("base64_data")

            content = json.dumps(
                {
                    "result": result,
                    "stdout": response.get("stdout"),
                    "stderr": response.get("stderr"),
                },
                indent=2,
            )
            return content, response
        finally:
            if self.delete_session_after_invocation:
                self.delete_session()

    def upload_file(
        self,
        *,
        data: Optional[BinaryIO] = None,
        remote_file_path: Optional[str] = None,
        local_file_path: Optional[str] = None,
    ) -> RemoteFileMetadata:
        """Upload a file to the session.

        Args:
            data: The data to upload. Requires `remote_file_path`.
            remote_file_path: The path to upload the file to, relative to
                `/mnt/data`. Required with `data`; if local_file_path is
                provided instead, this is defaulted to its filename.
            local_file_path: The path to the local file to upload.

        Returns:
            RemoteFileMetadata: The metadata for the uploaded file
        """
        api_url = self._build_url("files/upload")
        headers = self._auth_headers()

        with self._upload_payload(
            data=data,
            local_file_path=local_file_path,
            remote_file_path=remote_file_path,
        ) as (name, stream):
            files = [("file", (name, stream, "application/octet-stream"))]
            response = requests.request(
                "POST",
                api_url,
                headers=headers,
                data={},
                files=files,
                timeout=self.request_timeout,
            )
        response.raise_for_status()

        response_json = response.json()
        return RemoteFileMetadata.from_dict(response_json["value"][0])

    def download_file(
        self, *, remote_file_path: str, local_file_path: Optional[str] = None
    ) -> BinaryIO:
        """Download a file from the session.

        Args:
            remote_file_path: The path to download the file from,
                relative to `/mnt/data`.
            local_file_path: The path to save the downloaded file to.
                If not provided, the file is returned as a BufferedReader.

        Returns:
            BinaryIO: The data of the downloaded file.
        """
        encoded_remote_file_path = urllib.parse.quote(remote_file_path)
        api_url = self._build_url(f"files/content/{encoded_remote_file_path}")
        headers = self._auth_headers()

        response = requests.get(api_url, headers=headers, timeout=self.request_timeout)
        response.raise_for_status()

        if local_file_path:
            with open(local_file_path, "wb") as f:
                f.write(response.content)

        return BytesIO(response.content)

    def list_files(self) -> List[RemoteFileMetadata]:
        """List the files in the session.

        Returns:
            list[RemoteFileMetadata]: The metadata for the files in the session
        """
        api_url = self._build_url("files")
        headers = self._auth_headers()

        response = requests.get(api_url, headers=headers, timeout=self.request_timeout)
        response.raise_for_status()

        response_json = response.json()
        return [RemoteFileMetadata.from_dict(entry) for entry in response_json["value"]]


class SessionsBashTool(_BaseSessionsTool):
    r"""Azure Dynamic Sessions Bash tool for executing shell commands.

    **Setup:**

    ```bash
    pip install -U "langchain-azure-compute[dynamic-sessions]"
    ```

    ```python
    import getpass

    POOL_MANAGEMENT_ENDPOINT = getpass.getpass("Enter the management endpoint of the session pool: ")
    ```

    **Instantiation:**

    ```python

    from langchain_azure_compute.dynamic_sessions import SessionsBashTool

    tool = SessionsBashTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT
    )
    ```

    **Invocation with args:**

    ```python
    tool.invoke("echo hello")
    ```

    ```output
    '{\\n  "stdout": "hello\\n",\\n  "stderr": "",\\n  "exitCode": 0\\n}'
    ```

    **Invocation with ToolCall:**

    ```python
    tool.invoke({"args": {"input":"echo hello"}, "id": "1", "name": tool.name, "type": "tool_call"})
    ```

    ```output
    '{\\n  "stdout": "hello\\n",\\n  "stderr": "",\\n  "exitCode": 0\\n}'
    ```
    """  # noqa: E501

    name: str = "Bash"
    description: str = (
        "A Bash shell. Use this to execute bash commands "
        "when you need to perform file operations, run scripts, "
        "or execute system commands. "
        "Input should be a valid bash command. "
        "Returns a JSON object with the stdout, stderr, and exit code. "
    )

    _API_VERSION: ClassVar[str] = "2025-02-02-preview"

    def execute(self, bash_command: str) -> Any:
        """Execute a bash command in the session."""
        if self.sanitize_input:
            bash_command = _sanitize_bash_input(bash_command)

        api_url = self._build_url("executions")
        headers = self._auth_headers(json_body=True)
        body = {
            "shellCommand": bash_command,
        }

        response = requests.post(
            api_url, headers=headers, json=body, timeout=self.request_timeout
        )
        response.raise_for_status()
        response_json = response.json()
        return response_json

    def _run(self, bash_command: str, **kwargs: Any) -> Tuple[str, dict]:
        try:
            response = self.execute(bash_command)

            result = response.get("result", {})
            # The Shell session pool API returns exit code in the top-level "status"
            # field as a string (e.g. "0"), not in an "exitCode" field.
            status_raw = response.get("status")
            try:
                exit_code: Optional[int] = (
                    int(status_raw) if status_raw is not None else None
                )
            except (ValueError, TypeError):
                exit_code = None
            content = json.dumps(
                {
                    "stdout": result.get("stdout"),
                    "stderr": result.get("stderr"),
                    "exitCode": exit_code,
                },
                indent=2,
            )
            return content, response
        finally:
            if self.delete_session_after_invocation:
                self.delete_session()

    def upload_file(
        self,
        *,
        data: Optional[BinaryIO] = None,
        remote_file_path: Optional[str] = None,
        local_file_path: Optional[str] = None,
    ) -> RemoteFileMetadata:
        """Upload a file to the session.

        Args:
            data: The data to upload. Requires `remote_file_path`.
            remote_file_path: The path to upload the file to, relative to
                `/mnt/data`. Required with `data`; if local_file_path is
                provided instead, this is defaulted to its filename.
            local_file_path: The path to the local file to upload.

        Returns:
            RemoteFileMetadata: The metadata for the uploaded file
        """
        api_url = self._build_url("files", params={"path": "/mnt/data"})
        headers = self._auth_headers()

        with self._upload_payload(
            data=data,
            local_file_path=local_file_path,
            remote_file_path=remote_file_path,
        ) as (name, stream):
            files = [("file", (name, stream, "application/octet-stream"))]
            response = requests.request(
                "POST",
                api_url,
                headers=headers,
                data={},
                files=files,
                timeout=self.request_timeout,
            )
        response.raise_for_status()

        response_json = response.json()
        return RemoteFileMetadata(
            filename=response_json.get("name"),
            size_in_bytes=response_json.get("sizeInBytes"),
        )

    def download_file(
        self, *, remote_file_path: str, local_file_path: Optional[str] = None
    ) -> BinaryIO:
        """Download a file from the session.

        Args:
            remote_file_path: The path to download the file from,
                relative to `/mnt/data`.
            local_file_path: The path to save the downloaded file to.
                If not provided, the file is returned as a BufferedReader.

        Returns:
            BinaryIO: The data of the downloaded file.
        """
        encoded_remote_file_path = urllib.parse.quote(remote_file_path)
        api_url = self._build_url(f"files/{encoded_remote_file_path}/content")
        headers = self._auth_headers()

        response = requests.get(api_url, headers=headers, timeout=self.request_timeout)
        response.raise_for_status()

        if local_file_path:
            with open(local_file_path, "wb") as f:
                f.write(response.content)

        return BytesIO(response.content)

    def list_files(self) -> List[RemoteFileMetadata]:
        """List the files in the session.

        Returns:
            list[RemoteFileMetadata]: The metadata for the files in the session
        """
        api_url = self._build_url("files")
        headers = self._auth_headers()

        response = requests.get(api_url, headers=headers, timeout=self.request_timeout)
        response.raise_for_status()

        response_json = response.json()
        return [
            RemoteFileMetadata(
                filename=entry.get("name"),
                size_in_bytes=entry.get("sizeInBytes"),
            )
            for entry in response_json["value"]
        ]
