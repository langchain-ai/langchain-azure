"""Azure Dynamic Sessions backend for deepagents.

This module provides an ``SessionsBashBackend`` that implements the deepagents
``SandboxBackendProtocol`` by delegating shell execution, file upload, and
file download to the Azure Container Apps dynamic-sessions REST API.

Targets **Shell-typed** session pools: all file operations (``read``,
``write``, ``edit``, ``ls_info``, ``glob_info``) are overridden with
native bash commands instead of the ``python3 -c`` wrappers used by
``BaseSandbox``.  ``grep_raw`` is already bash-native in ``BaseSandbox``.
"""

from __future__ import annotations

import base64
import logging
import os
import shlex
import urllib.parse
from io import BytesIO
from typing import TYPE_CHECKING, Callable, Optional
from uuid import uuid4

import requests

if TYPE_CHECKING:
    from deepagents.backends.protocol import (
        EditResult,
        ExecuteResponse,
        FileDownloadResponse,
        FileInfo,
        FileUploadResponse,
        WriteResult,
    )
    from deepagents.backends.sandbox import BaseSandbox

from langchain_azure_dynamic_sessions._api.base import experimental
from langchain_azure_dynamic_sessions.tools.sessions import (
    USER_AGENT,
    _access_token_provider_factory,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120


@experimental()
class SessionsBashBackend(BaseSandbox):
    """Azure Dynamic Sessions backend for the deepagents framework.

    Extends ``BaseSandbox`` with bash-native overrides for all file
    operations (``read``, ``write``, ``edit``, ``ls_info``, ``glob_info``).
    This avoids the ``python3 -c`` wrappers from ``BaseSandbox``, which
    may not be available in Shell-typed session pools.

    Args:
        pool_management_endpoint: Management endpoint of the session pool.
        session_id: Identifier for the session. Defaults to a random UUID.
        access_token_provider: Callable returning an access token string.
            Defaults to ``DefaultAzureCredential`` targeting the
            ``https://dynamicsessions.io`` scope.
        timeout: Default timeout in seconds for shell commands.
        max_output_bytes: Maximum bytes captured from command output.
    """

    def __init__(
        self,
        pool_management_endpoint: str,
        *,
        session_id: Optional[str] = None,
        access_token_provider: Optional[Callable[[], Optional[str]]] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        max_output_bytes: int = 100_000,
    ) -> None:
        """Initialize an SessionsBashBackend.

        Args:
            pool_management_endpoint: The Azure Container Apps sessions pool
                management endpoint URL.
            session_id: Optional session identifier. A random UUID is generated
                if not provided.
            access_token_provider: Optional callable that returns a bearer
                token string. Uses ``DefaultAzureCredential`` by default.
            timeout: Default timeout in seconds for shell commands.
            max_output_bytes: Maximum bytes captured from command output.
        """
        self._pool_management_endpoint = pool_management_endpoint
        self._session_id = session_id or str(uuid4())
        self._access_token_provider = (
            access_token_provider or _access_token_provider_factory()
        )
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes
        logger.info(
            "SessionsBashBackend initialized (session_id=%s, endpoint=%s)",
            self._session_id,
            self._pool_management_endpoint,
        )

    # ------------------------------------------------------------------
    # URL helpers (mirrors SessionsBashTool._build_url)
    # ------------------------------------------------------------------

    def _build_url(self, path: str) -> str:
        endpoint = self._pool_management_endpoint
        if not endpoint:
            raise ValueError("pool_management_endpoint is not set")
        if not endpoint.endswith("/"):
            endpoint += "/"
        encoded_session_id = urllib.parse.quote(self._session_id)
        query = f"identifier={encoded_session_id}&api-version=2025-02-02-preview"
        query_separator = "&" if "?" in endpoint else "?"
        return endpoint + path + query_separator + query

    def _auth_headers(self) -> dict[str, str]:
        token = self._access_token_provider()
        return {
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
        }

    # ------------------------------------------------------------------
    # SandboxBackendProtocol — abstract members
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """Unique identifier for this backend instance."""
        return self._session_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command in the Azure Dynamic Sessions sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum seconds to wait. Falls back to the instance
                default if *None*.

        Returns:
            ``ExecuteResponse`` with combined stdout/stderr, exit code,
            and truncation flag.
        """
        effective_timeout = timeout if timeout is not None else self._timeout

        api_url = self._build_url("executions")
        headers = {**self._auth_headers(), "Content-Type": "application/json"}
        body = {
            "shellCommand": command,
        }

        cmd_preview = command[:120] + ("..." if len(command) > 120 else "")
        logger.debug(
            "execute: POST %s (timeout=%s, command=%r)",
            api_url,
            effective_timeout,
            cmd_preview,
        )
        logger.debug("execute: request body=%s", body)

        response = requests.post(
            api_url,
            headers=headers,
            json=body,
            timeout=effective_timeout,
        )

        logger.debug(
            "execute: response status=%s length=%s",
            response.status_code,
            len(response.content),
        )

        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            logger.error(
                "execute: request failed (%s): %s",
                response.status_code,
                detail,
            )
            raise RuntimeError(
                f"Execution request failed ({response.status_code}): {detail}"
            )

        data = response.json()
        logger.debug("execute: response data=%s", data)

        result = data.get("result", {})
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        output = stdout + stderr

        truncated = False
        if len(output.encode("utf-8")) > self._max_output_bytes:
            output = output[: self._max_output_bytes]
            truncated = True

        raw_status = data.get("status")
        exit_code = int(raw_status) if raw_status is not None else None

        exec_response = ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=truncated,
        )
        logger.debug(
            "execute: exit_code=%s truncated=%s output_len=%d",
            exec_response.exit_code,
            exec_response.truncated,
            len(exec_response.output),
        )
        return exec_response

    # ------------------------------------------------------------------
    # Bash-native file operations (override BaseSandbox python3 wrappers)
    # ------------------------------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:
        """List directory contents using ``ls``."""
        cmd = f"ls -1apL {shlex.quote(path)} 2>/dev/null || true"
        result = self.execute(cmd)
        items: list[FileInfo] = []
        for name in result.output.strip().splitlines():
            if not name or name in ("./", "../"):
                continue
            is_dir = name.endswith("/")
            items.append({"path": name.rstrip("/"), "is_dir": is_dir})
        return items

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file with line numbers using ``awk``."""
        start = offset + 1
        end = offset + limit
        cmd = (
            f"awk 'NR>={start} && NR<={end} "
            f'{{printf "%6d\\t%s\\n", NR, $0}}\' '
            f"{shlex.quote(file_path)} 2>&1"
        )
        result = self.execute(cmd)
        if result.exit_code != 0 or "No such file" in result.output:
            return f"Error: File '{file_path}' not found"
        return result.output.rstrip()

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file using ``base64`` decode + shell redirect."""
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        escaped_path = shlex.quote(file_path)
        cmd = (
            f"if [ -e {escaped_path} ]; then "
            f"echo \"Error: File '{file_path}' already exists\" >&2; exit 1; fi && "
            f"mkdir -p $(dirname {escaped_path}) && "
            f"echo '{b64}' | base64 -d > {escaped_path}"
        )
        result = self.execute(cmd)
        if result.exit_code != 0 or "Error:" in result.output:
            msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=msg)
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file using ``awk`` for reliable string replacement."""
        old_b64 = base64.b64encode(old_string.encode("utf-8")).decode("ascii")
        new_b64 = base64.b64encode(new_string.encode("utf-8")).decode("ascii")
        escaped_path = shlex.quote(file_path)
        ra_flag = "1" if replace_all else "0"
        cmd = (
            f"OLD=$(echo '{old_b64}' | base64 -d) && "
            f"NEW=$(echo '{new_b64}' | base64 -d) && "
            f"if [ ! -f {escaped_path} ]; then exit 3; fi && "
            f'COUNT=$(grep -cF "$OLD" {escaped_path} || true) && '
            f'if [ "$COUNT" -eq 0 ]; then exit 1; fi && '
            f'if [ "$COUNT" -gt 1 ] && [ "{ra_flag}" = "0" ]; then '
            f"echo $COUNT; exit 2; fi && "
            f"TMPF=$(mktemp) && "
            f'awk -v old="$OLD" -v new="$NEW" -v ra={ra_flag} '
            "'BEGIN{done=0} "
            "{line=$0; "
            "while(idx=index(line,old)) { "
            "if(!ra && done){break}; "
            'printf "%s%s",substr(line,1,idx-1),new; '
            "line=substr(line,idx+length(old)); done=1}; "
            "print line}' "
            f'{escaped_path} > "$TMPF" && '
            f'mv "$TMPF" {escaped_path} && '
            f"echo $COUNT"
        )
        result = self.execute(cmd)
        if result.exit_code == 1:
            return EditResult(error=f"Error: String not found in file: '{old_string}'")
        if result.exit_code == 2:
            return EditResult(
                error=f"Error: String '{old_string}' appears multiple times. "
                "Use replace_all=True to replace all occurrences."
            )
        if result.exit_code == 3:
            return EditResult(error=f"Error: File '{file_path}' not found")
        if result.exit_code != 0:
            return EditResult(
                error=result.output.strip() or f"Edit failed for '{file_path}'"
            )
        try:
            count = int(result.output.strip())
        except ValueError:
            count = 1
        return EditResult(path=file_path, files_update=None, occurrences=count)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Glob matching using ``find``."""
        escaped_path = shlex.quote(path)
        cmd = (
            f"cd {escaped_path} 2>/dev/null && "
            f"find . -path './{pattern}' -printf '%P\\n' 2>/dev/null | "
            "while IFS= read -r f; do "
            'if [ -d "$f" ]; then echo "D:$f"; else echo "F:$f"; fi; done'
        )
        result = self.execute(cmd)
        items: list[FileInfo] = []
        for line in result.output.strip().splitlines():
            if not line or ":" not in line:
                continue
            kind, fpath = line.split(":", 1)
            items.append({"path": fpath, "is_dir": kind == "D"})
        return items

    # ------------------------------------------------------------------
    # File transfer via REST API
    # ------------------------------------------------------------------

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """Upload files to the session's ``/mnt/data`` directory.

        Args:
            files: List of ``(path, content)`` tuples.

        Returns:
            One ``FileUploadResponse`` per input file.  Errors are captured
            per-file rather than raised.
        """
        responses: list[FileUploadResponse] = []
        api_url = self._build_url("files") + "&path=/mnt/data"
        headers = self._auth_headers()
        logger.debug("upload_files: %d file(s) to %s", len(files), api_url)

        for path, content in files:
            try:
                filename = os.path.basename(path)
                logger.debug(
                    "upload_files: uploading %s (%d bytes)", path, len(content)
                )
                multipart = [
                    ("file", (filename, BytesIO(content), "application/octet-stream"))
                ]
                resp = requests.post(
                    api_url,
                    headers=headers,
                    data={},
                    files=multipart,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                logger.debug("upload_files: %s succeeded (%s)", path, resp.status_code)
                responses.append(FileUploadResponse(path=path))
            except Exception:
                logger.warning("upload_files: %s failed", path, exc_info=True)
                responses.append(
                    FileUploadResponse(path=path, error="permission_denied")
                )
        return responses

    def download_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        """Download files from the session's ``/mnt/data`` directory.

        Args:
            paths: List of file paths to download.

        Returns:
            One ``FileDownloadResponse`` per input path.  Errors are captured
            per-file rather than raised.
        """
        responses: list[FileDownloadResponse] = []
        headers = self._auth_headers()
        logger.debug("download_files: %d file(s)", len(paths))

        for path in paths:
            try:
                encoded = urllib.parse.quote(os.path.basename(path))
                api_url = self._build_url(f"files/{encoded}/content")
                logger.debug("download_files: GET %s", api_url)
                resp = requests.get(api_url, headers=headers, timeout=self._timeout)
                resp.raise_for_status()
                logger.debug(
                    "download_files: %s succeeded (%d bytes)",
                    path,
                    len(resp.content),
                )
                responses.append(FileDownloadResponse(path=path, content=resp.content))
            except Exception:
                logger.warning("download_files: %s failed", path, exc_info=True)
                responses.append(
                    FileDownloadResponse(path=path, error="file_not_found")
                )
        return responses
