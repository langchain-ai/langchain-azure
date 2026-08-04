"""Azure Dynamic Sessions backend for deepagents.

This module provides an ``SessionsBashBackend`` that implements the deepagents
``SandboxBackendProtocol`` by delegating shell execution, file upload, and
file download to the Azure Container Apps dynamic-sessions REST API.

Targets **Shell-typed** session pools: the file operations (``read``,
``write``, ``edit``, ``ls``, ``glob``, ``grep``) are overridden with native
shell commands instead of the ``python3 -c`` wrappers used by ``BaseSandbox``,
which a shell-only pool image may not be able to run. The async variants
delegate to these sync overrides in a worker thread: ``BaseSandbox``'s own
async methods rebuild its ``python3``-based commands rather than calling the
sync overrides, so inheriting them would silently bypass everything above.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import logging
import posixpath
import urllib.parse
from io import BytesIO
from typing import TYPE_CHECKING, Mapping
from uuid import uuid4

import requests
from langchain_core._api import beta

# Check `deepagents` itself rather than catching ImportError from the import
# below: that would also swallow an unrelated ImportError (e.g. a circular
# import bug) and report it as a missing extra.
if importlib.util.find_spec("deepagents") is None:
    raise ImportError(
        "SessionsBashBackend requires deepagents, which is provided by the "
        "'dynamic-sessions' extra. Install it with: pip install "
        "'langchain-azure-container-apps[dynamic-sessions]' "
        "(requires Python >= 3.11)."
    )

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    PERMISSION_DENIED,
    EditResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import ASYNC_GREP_TIMEOUT, TRUNCATION_MSG, BaseSandbox

if TYPE_CHECKING:
    from typing import Callable, Optional

    from deepagents.backends.protocol import FileInfo, GrepMatch

from langchain_azure_container_apps.dynamic_sessions._common import (
    access_token_provider_factory,
    auth_headers,
    build_session_url,
)
from langchain_azure_container_apps.dynamic_sessions.backends import _shell

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120

_API_VERSION = "2025-02-02-preview"

_BETA_MESSAGE = (
    "`SessionsBashBackend` is in public preview. "
    "Its API is not stable and may change in future versions."
)

# The dynamic-sessions file API is a flat store rooted here: uploads take a
# bare filename and downloads address one by name. There is no addressable
# subdirectory, so this is the only path shape the transfer methods can honour.
_FILE_ROOT = "/mnt/data"

# Returned when the data plane dropped a command's output on every attempt.
# Says what to do rather than naming a protocol code: borrowing one would send
# the agent down a recovery path for a condition that is not what happened.
_OUTPUT_LOST = "the session returned no output for this command; retry it"

# Names the knob, not `limit`: a smaller window cannot shrink the read
# command's own bookkeeping, so shrinking the request would not help.
_READ_CAP_TOO_SMALL = (
    "max_output_bytes is too small to carry the chunked read transport"
)

# Longest failure detail reflected into a result error. Anything past this is
# noise to the model and could be an entire file's content on a garbled read.
_MAX_ERROR_DETAIL = 200

# The dynamic-sessions data plane caps each of stdout and stderr at exactly
# this many bytes per execution, silently: status stays "0" and no response
# field marks the cut (measured live, deterministic at the exact byte).
# `read` pages under it via base64 chunks; `execute` flags an at-cap stream
# as truncated; `ls`/`glob`/`grep` route through that flag into their
# designed truncation handling. Defined in `_shell`, which sizes read chunks
# against it -- one number, so the two cannot drift apart.
_SERVICE_OUTPUT_CAP = _shell.SERVICE_OUTPUT_CAP


def _request_timeout(value: float) -> float | None:
    """Translate a protocol timeout into Requests form.

    The protocol lets 0 disable the timeout; Requests rejects 0 outright
    ("cannot be set to a value less than or equal to 0") and spells no-timeout
    as None. Every `requests` call in this module must go through this --
    forwarding 0 raw raises a bare `ValueError`, which is not a
    `RequestException` and so would escape the per-file error capture in the
    transfer methods.
    """
    return value if value > 0 else None


def _session_file_name(path: str) -> str | None:
    """Return the stored filename for `path`, or `None` if it is not storable.

    Anything outside the flat `/mnt/data` root is rejected rather than silently
    flattened into it: reporting the caller's path as uploaded while writing
    somewhere else makes a later `read()` of that path fail for no visible
    reason, and collapses `/a/f.txt` and `/b/f.txt` onto each other.

    `normpath` first, so `/mnt/data/../etc/passwd` cannot traverse out and
    `/mnt/database` cannot pass as a prefix match.
    """
    normalized = posixpath.normpath(path)
    prefix = _FILE_ROOT + "/"
    if not normalized.startswith(prefix):
        return None
    name = normalized[len(prefix) :]
    if not name or "/" in name:
        return None
    return name


def _bounded(detail: str) -> str:
    """Cap a failure detail destined for a result error string."""
    detail = detail.strip()
    if len(detail) > _MAX_ERROR_DETAIL:
        return detail[:_MAX_ERROR_DETAIL] + "..."
    return detail


def _transfer_error(exc: Exception, *, upload: bool) -> str:
    """Classify a file-transfer failure for `FileUploadResponse.error`.

    Only statuses that map cleanly become protocol codes; anything else gets a
    bounded description, because a wrong code sends the agent down the wrong
    recovery path. A 404 is only `file_not_found` on download: the protocol
    documents that code for missing *files*, while an upload 404 means the
    session or pool route itself resolved nowhere.

    Do not interpolate the exception: `requests` embeds the request URL in its
    messages, which carries the session identifier and the pool management
    endpoint, and this string is handed to the model.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return PERMISSION_DENIED
    if status == 404 and not upload:
        return FILE_NOT_FOUND
    if status is not None:
        return f"request failed with HTTP {status}"
    return f"request failed: {type(exc).__name__}"


@beta(message=_BETA_MESSAGE)
class SessionsBashBackend(BaseSandbox):
    """Azure Dynamic Sessions backend for the deepagents framework.

    Extends ``BaseSandbox`` with shell-native overrides for all file
    operations (``read``, ``write``, ``edit``, ``ls``, ``glob``, ``grep``) and
    async variants that delegate to them. This avoids the ``python3 -c``
    wrappers from ``BaseSandbox``, which may not be available in Shell-typed
    session pools.

    One deliberate divergence from the deepagents 0.7 filesystem middleware's
    tool description: ``write`` refuses to overwrite an existing file, matching
    the shared ``langchain-tests`` sandbox suite (and the sibling Azure Blob
    backend) rather than the middleware's "replaces it entirely" wording. The
    error tells the model how to proceed.

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
        """Initialize a SessionsBashBackend.

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
            access_token_provider or access_token_provider_factory()
        )
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes
        logger.info(
            "SessionsBashBackend initialized (session_id=%s, endpoint=%s)",
            self._session_id,
            self._pool_management_endpoint,
        )

    def _build_url(
        self, resource: str, *, params: Mapping[str, str] | None = None
    ) -> str:
        return build_session_url(
            self._pool_management_endpoint,
            resource,
            session_id=self._session_id,
            api_version=_API_VERSION,
            params=params,
        )

    def _auth_headers(self) -> dict[str, str]:
        return auth_headers(self._access_token_provider())

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
            and truncation flag. ``exit_code`` is ``None`` when the service
            returned no status or a non-numeric one (a Python-typed pool
            reports states like ``"Succeeded"`` -- this backend requires a
            Shell-typed pool).

        Raises:
            RuntimeError: If the service rejects the execution request.
            requests.RequestException: On a transport-level failure.
        """
        effective_timeout = timeout if timeout is not None else self._timeout
        request_timeout = _request_timeout(effective_timeout)

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
        response = requests.post(
            api_url,
            headers=headers,
            json=body,
            timeout=request_timeout,
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

        result = data.get("result", {})
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        output = stdout + stderr

        # The data plane truncates each stream at exactly _SERVICE_OUTPUT_CAP
        # bytes and signals nothing: the status stays "0" and no response
        # field marks the loss (measured live; deterministic). A stream
        # arriving at the cap is therefore flagged as truncated here, with a
        # three-byte margin because a multi-byte character cut at the
        # boundary is replaced before it reaches this client. An output of
        # legitimately exactly this size gains a spurious flag; that is the
        # honest direction to be wrong in.
        truncated = any(
            len(stream.encode("utf-8")) >= _SERVICE_OUTPUT_CAP - 3
            for stream in (stdout, stderr)
        )
        encoded = output.encode("utf-8")
        if len(encoded) > self._max_output_bytes:
            # Slice the encoded bytes, not characters: a character-based slice
            # leaves multi-byte output over the cap it just measured.
            output = encoded[: self._max_output_bytes].decode("utf-8", "ignore")
            truncated = True

        raw_status = data.get("status")
        try:
            exit_code = int(raw_status) if raw_status is not None else None
        except (ValueError, TypeError):
            # A Python-typed pool reports lifecycle states ("Succeeded") here;
            # surface "no usable exit code" rather than crashing mid-parse.
            logger.warning("execute: non-numeric status %r", raw_status)
            exit_code = None

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

    def _run_program(self, program: str) -> tuple[ExecuteResponse, str] | None:
        """Run an idempotent generated program and strip its completion marker.

        Returns `(result, output)`, or `None` when the data plane lost the
        command's final write on every attempt -- see `_shell.with_done`.

        Idempotent only: a lost tail is retried once, and re-running a mutation
        whose first attempt actually succeeded would report `already_exists` or
        `not found` against its own effect. `write` and `edit` therefore
        bypass this helper and ride on the exit status instead.

        A truncated response is missing its tail for a known reason and is
        returned as-is -- minus any marker fragment the cap cut in half --
        for the caller to handle per `result.truncated`.
        """
        for attempt in range(2):
            done = _shell.make_done_token()
            result = self.execute(_shell.with_done(program, done))
            output, complete = _shell.split_done(result.output, done)
            if result.truncated and not complete:
                output = _shell.strip_partial_done(output, done)
            if complete or result.truncated:
                return result, output
            logger.warning(
                "output tail lost on attempt %d: the data plane dropped the "
                "command's final write",
                attempt + 1,
            )
        return None

    def ls(self, path: str) -> LsResult:
        """List directory contents.

        Entries carry absolute paths, matching `BaseSandbox.ls`.

        A listing that exceeds `max_output_bytes` is reported as an error:
        `LsResult` has no truncation flag, and presenting the surviving prefix
        as the whole directory would hide entries from the agent with no
        indication anything is missing.
        """
        run = self._run_program(_shell.build_ls_command(path))
        if run is None:
            return LsResult(entries=None, error=f"Path '{path}': {_OUTPUT_LOST}")
        result, output = run
        # `split`, not `strip().splitlines()`: a leading- or trailing-space
        # filename sorts to an edge of the listing, where `strip` would eat the
        # space and name a different, usually nonexistent, file.
        lines = [line for line in output.split("\n") if line]
        # The probes print a sentinel and stop before `ls` runs, so a
        # legitimate sentinel can only ever be the first line; checking later
        # lines would let a file literally named `__ERR__...` poison the
        # listing for its whole directory.
        if lines:
            code = _shell.error_code(lines[0])
            if code is not None:
                return LsResult(entries=None, error=f"Path '{path}': {code}")
        if result.truncated:
            return LsResult(
                entries=None,
                error=(
                    f"Path '{path}': listing exceeded the output cap; "
                    "list a subdirectory instead"
                ),
            )
        entries: list[FileInfo] = []
        for name in lines:
            if name in ("./", "../"):
                continue
            is_dir = name.endswith("/")
            entries.append(
                {
                    "path": posixpath.join(path, name.rstrip("/")),
                    "is_dir": is_dir,
                }
            )
        return LsResult(entries=entries)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read a slice of a text file.

        The window is fetched in base64 chunks sized under the data plane's
        4,096-byte output cap (`_shell.SERVICE_OUTPUT_CAP`): the first command
        returns the line count, the window's byte size, and the first chunk;
        follow-up commands page through the rest. One round trip for windows
        under ~3 KB, more for larger ones. The chunks are separate executions,
        so a file rewritten mid-read can tear; a *shrinking* window is
        detected and reported rather than stitched.

        CRLF line endings are normalized to LF, matching `BaseSandbox.read`,
        so returned content round-trips through `edit`'s exact matching.

        Binary files are not detected: unlike `BaseSandbox.read`, everything is
        returned as utf-8 text with undecodable bytes replaced.

        A window that exceeds `max_output_bytes` is fetched only up to the
        cap, cut back to whole lines, the truncation is called out in the
        content, and `next_offset` resumes at the first line not fully shown.

        `max_output_bytes` caps the *content* returned, but `execute` also
        applies it to the wire, so the chunks are sized against whichever of
        it and the service cap is smaller -- a small cap costs round trips,
        never a corrupted record. Below `_shell.READ_BOOKKEEPING_BYTES` no
        chunk fits at all and the read reports that rather than failing to
        decode.
        """
        if limit <= 0:
            return ReadResult(
                file_data=FileData(content="", encoding="utf-8"),
                no_lines_requested=True,
            )
        offset = max(offset, 0)
        start = offset + 1
        end = offset + limit
        wire = min(self._max_output_bytes, _SERVICE_OUTPUT_CAP)
        take = min(
            _shell.READ_CHUNK_BYTES,
            _shell.chunk_for_wire(wire - _shell.READ_BOOKKEEPING_BYTES),
        )
        if take <= 0:
            return ReadResult(error=f"File '{file_path}': {_READ_CAP_TOO_SMALL}")
        run = self._run_program(
            _shell.build_read_command(file_path, start=start, end=end, take=take)
        )
        if run is None:
            return ReadResult(error=f"File '{file_path}': {_OUTPUT_LOST}")
        result, output = run

        code = _shell.error_code(output.split("\n", 1)[0])
        if code is not None:
            return ReadResult(error=f"File '{file_path}': {code}")
        if result.exit_code != 0:
            detail = _bounded(output) or f"exit code {result.exit_code}"
            return ReadResult(error=f"File '{file_path}': {detail}")
        # Exactly three records: the counts are digits and the chunk is
        # base64, so neither can contain the separator -- file content cannot
        # forge extra records the way raw text could.
        parts = output.split(f"{_shell.SEP_SENTINEL}\n")
        if len(parts) != 3:
            if result.truncated:
                # Defensive: the chunk sizing above keeps the wire under the
                # cap, so a truncated response here means the bookkeeping
                # itself overran the margin. A smaller `limit` cannot shrink
                # that, so the advice names the real knob.
                return ReadResult(error=f"File '{file_path}': {_READ_CAP_TOO_SMALL}")
            return ReadResult(
                error=(
                    f"File '{file_path}': unexpected read output: {_bounded(output)!r}"
                )
            )
        head, size_part, chunk_part = parts
        try:
            total_lines = int(head.strip())
        except ValueError:
            return ReadResult(
                error=f"File '{file_path}': could not determine line count"
            )
        try:
            window_bytes = int(size_part.strip())
        except ValueError:
            return ReadResult(
                error=f"File '{file_path}': could not determine window size"
            )

        if total_lines == 0:
            return ReadResult(
                file_data=FileData(
                    content="System reminder: File exists but has empty contents",
                    encoding="utf-8",
                )
            )
        if offset >= total_lines:
            return ReadResult(
                error=(
                    f"File '{file_path}': Line offset {offset} exceeds file length "
                    f"({total_lines} lines)"
                )
            )

        try:
            data = base64.b64decode(chunk_part.strip(), validate=True)
        except ValueError:
            return ReadResult(
                error=(
                    f"File '{file_path}': unexpected read output: "
                    f"{_bounded(chunk_part)!r}"
                )
            )
        # Fetch to the window size or the client cap, whichever ends first;
        # past the cap the page is reported truncated exactly as before.
        goal = min(window_bytes, self._max_output_bytes)
        while len(data) < goal:
            chunk_run = self._run_program(
                _shell.build_read_chunk_command(
                    file_path,
                    start=start,
                    end=end,
                    skip=len(data),
                    take=min(take, goal - len(data)),
                )
            )
            if chunk_run is None:
                return ReadResult(error=f"File '{file_path}': {_OUTPUT_LOST}")
            _, chunk_output = chunk_run
            code = _shell.error_code(chunk_output.split("\n", 1)[0])
            if code is not None:
                return ReadResult(error=f"File '{file_path}': {code}")
            try:
                chunk = base64.b64decode(chunk_output.strip(), validate=True)
            except ValueError:
                return ReadResult(
                    error=(
                        f"File '{file_path}': unexpected read output: "
                        f"{_bounded(chunk_output)!r}"
                    )
                )
            if not chunk:
                # The measured window has fewer bytes than it did one command
                # ago: the file changed under the read.
                return ReadResult(error=f"File '{file_path}': file changed during read")
            data += chunk

        truncated = window_bytes > self._max_output_bytes
        body = data[:goal].decode("utf-8", "replace")

        truncation_note = ""
        if truncated:
            # The cap almost certainly landed mid-line. Keep only fully
            # rendered lines (each followed by "\n") so the partial tail is
            # neither shown nor skipped: `next_offset` re-reads it from its
            # start. Without this the boundary line's tail would be silently
            # unrecoverable -- every later page starts after it.
            #
            # When no full line rendered, the first requested line alone
            # overflows the cap: keep the partial head, which counts as one
            # returned line below, so `next_offset` advances past a line whose
            # tail no limit could ever render -- `BaseSandbox.read` makes the
            # same one-line advance rather than erroring on every retry.
            rendered = body[: body.rfind("\n") + 1]
            if rendered:
                body = rendered
            truncation_note = TRUNCATION_MSG

        content = body[:-1] if body.endswith("\n") else body
        # Normalize CRLF: the window is cut by line number remotely, so this
        # only rewrites line endings, never the pagination bookkeeping. A file
        # using bare CR has no line breaks awk recognizes and passes through
        # unchanged.
        content = content.replace("\r\n", "\n")
        if content.endswith("\r"):
            content = content[:-1]
        # Count from the body, not the stripped content: a window of "\n" is
        # one blank line, and counting "" as zero lines yields end_line=0
        # against start_line=1, which the protocol rejects outright.
        returned_lines = content.count("\n") + 1 if body else 0
        if returned_lines == 0:
            return ReadResult(
                error=f"File '{file_path}': read returned no lines for a "
                f"{total_lines}-line file"
            )
        end_line = min(offset + returned_lines, total_lines)
        next_offset = end_line if end_line < total_lines else None
        return ReadResult(
            file_data=FileData(content=content + truncation_note, encoding="utf-8"),
            total_lines=total_lines,
            start_line=start,
            end_line=end_line,
            next_offset=next_offset,
        )

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file, refusing to overwrite an existing one.

        The refusal matches the shared `langchain-tests` sandbox suite, not the
        deepagents middleware's tool description (which promises overwrite) --
        see the class docstring. The error tells the model what to do instead.

        Deliberately not guarded by a completion marker like `ls`/`read`/`glob`:
        the outcome rides entirely on the exit status, which the data plane
        returns in its own field and so cannot lose with the output. Gating on
        the marker would fail writes that had already succeeded.

        Content above `_shell.MAX_TRANSPORT_BYTES` (~90 KB) exceeds what one
        shell command can carry and is refused with an actionable error;
        `BaseSandbox` routes large writes through `upload_files`, which here
        is limited to the flat `/mnt/data` store.
        """
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > _shell.MAX_TRANSPORT_BYTES:
            return WriteResult(
                error=(
                    f"Error: content is {content_bytes} bytes; the shell "
                    f"transport supports about {_shell.MAX_TRANSPORT_BYTES}. "
                    "Split the content across smaller files, or upload to "
                    "/mnt/data via upload_files."
                )
            )
        result = self.execute(_shell.build_write_command(file_path, content))
        if result.exit_code == _shell.WRITE_EXISTS:
            return WriteResult(
                error=(
                    f"File '{file_path}' already exists. Use edit to modify "
                    "it, or delete it first."
                )
            )
        if result.exit_code != 0:
            detail = _bounded(result.output) or f"exit code {result.exit_code}"
            return WriteResult(error=f"Failed to write file '{file_path}': {detail}")
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace an exact substring in a file.

        Matching is over the whole file, so `old_string` may span newlines and
        two occurrences on one line count as two. `old_string` is tried as
        sent, then as CRLF, then as LF, so content pasted back from `read`
        (which normalizes to LF) matches a CRLF file and the file keeps its own
        line endings -- the same three-way match `BaseSandbox.edit` performs.

        The outcome rides on the exit status and cannot be lost with the
        output, so the edit itself is never retried; the occurrence count
        travels on stderr, though, and on the rare lost-output run it is
        reported as `None` rather than guessed.

        `old_string` and `new_string` are budgeted *together* against
        `_shell.MAX_TRANSPORT_BYTES` (~90 KB): both base64 literals ride in one
        `sh -c` argument, so it is their sum the transport has to carry. Over
        that, the edit is refused with an actionable error.
        """
        # Summed, not maxed: two individually legal payloads still share one
        # argv string, and 50 KB + 50 KB exceeds MAX_ARG_STRLEN on its own.
        combined = len(old_string.encode("utf-8")) + len(new_string.encode("utf-8"))
        if combined > _shell.MAX_TRANSPORT_BYTES:
            return EditResult(
                error=(
                    f"Error: old_string and new_string total {combined} bytes; "
                    "the shell transport carries both in one command and "
                    f"supports about {_shell.MAX_TRANSPORT_BYTES}. Make the "
                    "edit in smaller pieces."
                )
            )
        done = _shell.make_done_token()
        result = self.execute(
            _shell.with_done(
                _shell.build_edit_command(
                    file_path, old_string, new_string, replace_all=replace_all
                ),
                done,
            )
        )
        output, complete = _shell.split_done(result.output, done)
        if result.exit_code == _shell.EDIT_NOT_FOUND:
            return EditResult(error=f"Error: String not found in file: '{old_string}'")
        if result.exit_code == _shell.EDIT_AMBIGUOUS:
            return EditResult(
                error=f"Error: String '{old_string}' appears multiple times. "
                "Use replace_all=True to replace all occurrences."
            )
        if result.exit_code == _shell.EDIT_MISSING_FILE:
            return EditResult(error=f"Error: File '{file_path}' not found")
        if result.exit_code == _shell.EDIT_EMPTY_OLD:
            return EditResult(error="Error: old_string must not be empty")
        if result.exit_code != 0:
            return EditResult(
                error=_bounded(output) or f"Edit failed for '{file_path}'"
            )
        count: int | None
        try:
            count = int(output.strip())
        except ValueError:
            # The count is the program's only stdout/stderr payload; if the
            # marker is also missing, the data plane dropped the tail and the
            # count is unknowable -- report that rather than inventing one.
            count = None
            if complete:
                logger.warning("edit: unparsable occurrence count %r", output[:80])
        return EditResult(path=file_path, occurrences=count)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Glob matching using ``find``.

        Matches are relative to `path`, matching `BaseSandbox.glob`, with the
        protocol's Python-glob semantics for the common pattern shapes: `*`
        stays within one path segment, `**/name` matches at any depth, a
        trailing `/` matches directories only. See `_shell._glob_invocations`
        for the exact translation and its documented approximations (a
        wildcard may still cross `/` inside a `HEAD/**/TAIL` pattern or one
        with multiple `**`; `.` segments are honored for matching but results
        come back in canonical spelling, so `./x.txt` reports `x.txt`).

        A `..` segment is rejected outright, so a pattern cannot walk out of
        `path`.

        Note the default `path` is `/`: on a session image that means walking
        the entire filesystem. Pass `/mnt/data` (or wherever the files of
        interest live) for anything latency-sensitive.
        """
        search_path = path or "/"
        rel_pattern = pattern.lstrip("/")
        if _shell.has_traversal(rel_pattern):
            return GlobResult(
                matches=None, error=f"Path '{search_path}': invalid_pattern"
            )
        run = self._run_program(_shell.build_glob_command(search_path, rel_pattern))
        if run is None:
            return GlobResult(
                matches=None, error=f"Path '{search_path}': {_OUTPUT_LOST}"
            )
        result, output = run
        # The probes report a missing, non-directory, or unreadable root with
        # the same codes `ls` uses, printed as the first line -- and only the
        # first line can be a legitimate sentinel, since the probes stop the
        # program before `find` runs.
        first_line = output.split("\n", 1)[0]
        code = _shell.error_code(first_line)
        if code is not None:
            return GlobResult(matches=None, error=f"Path '{search_path}': {code}")
        # After the probes, a nonzero status means the root vanished between
        # probe and `cd` -- report it as missing rather than as an empty match
        # set, which would tell the agent the directory exists. A `None` exit
        # code is its own condition (the service returned no status).
        if result.exit_code is None:
            return GlobResult(
                matches=None,
                error=f"Path '{search_path}': command returned no exit status",
            )
        if result.exit_code != 0:
            return GlobResult(
                matches=None, error=f"Path '{search_path}': path_not_found"
            )
        lines = output.split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # the terminator of the final record, not a match
        if result.truncated and lines and not output.endswith("\n"):
            # The cap cut the final line mid-name; a fabricated partial entry
            # (`par` for `part.txt`) is worse than one fewer match. A final
            # newline means the last line survived whole.
            lines = lines[:-1]
        matches: list[FileInfo] = []
        for line in lines:
            if not line or ":" not in line:
                continue
            kind, fpath = line.split(":", 1)
            matches.append({"path": fpath, "is_dir": kind == "D"})
        return GlobResult(matches=matches, truncated=result.truncated)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search file contents for a literal string using `grep -F`.

        Overridden for two reasons: `BaseSandbox.grep` routes slash-containing
        globs through a `python3 -c` wrapper a Shell pool may not run, and its
        command is not protected by the completion marker -- at the measured
        1.6-4% output-loss rate the final matches of a search would silently
        vanish. `grep` is idempotent, so the marker's retry is safe here.

        Args:
            pattern: Literal string to search for (not a regex).
            path: Directory or file to search in. Defaults to `"."`.
            glob: Optional glob restricting which files are searched. Patterns
                without a `/` match basenames at any depth via
                `grep --include`; patterns with a `/` match the
                search-root-relative path via the same `find` translation
                `glob` uses.
            max_count: Optional total cap on returned matches. `None` returns
                every match; an int stops the search at the cap and flags the
                result `truncated=True`.

        Returns:
            `GrepResult` with `GrepMatch` dicts, or `error` on failure.
        """
        root = path or "."
        run = self._run_program(
            _shell.build_grep_command(
                root, pattern, include_glob=glob, max_count=max_count
            )
        )
        if run is None:
            return GrepResult(error=f"Path '{root}': {_OUTPUT_LOST}")
        result, output = run
        ended_whole = output.endswith("\n")
        output = output.rstrip("\n")
        lines = output.split("\n") if output else []
        if lines:
            code = _shell.error_code(lines[0])
            if code is not None:
                return GrepResult(error=f"Path '{root}': {code}")
        if result.exit_code is not None and result.exit_code != 0:
            return GrepResult(
                error=f"Path '{root}': {_bounded(output) or 'search failed'}"
            )
        if result.truncated and lines and not ended_whole:
            # The byte cap cut the last record mid-line; drop it rather than
            # hand the model a half match.
            lines = lines[:-1]
        matches: list[GrepMatch] = []
        parse_error: str | None = None
        for line in lines:
            # Records are `path NUL line_number:text`; the find route emits
            # paths relative to the root it `cd`'d into.
            try:
                file_path, rest = line.split("\0", 1)
                line_num_str, text = rest.split(":", 1)
            except ValueError:
                parse_error = line
                continue
            if file_path.startswith("./") and root != ".":
                file_path = posixpath.join(root, file_path[2:])
            try:
                matches.append(
                    {"path": file_path, "line": int(line_num_str), "text": text}
                )
            except ValueError:
                parse_error = line
        if parse_error is not None and not matches:
            return GrepResult(error=f"Path '{root}': {_bounded(parse_error)}")
        if max_count is not None and len(matches) > max_count:
            return GrepResult(matches=matches[:max_count], truncated=True)
        return GrepResult(matches=matches, truncated=result.truncated)

    # ------------------------------------------------------------- async ----
    # `BaseSandbox`'s async variants rebuild its own `python3 -c` commands and
    # send them through `aexecute`, bypassing every override above -- including
    # the `/mnt/data` transfer rules and the output-loss compensation. Each
    # delegates to its sync counterpart in a worker thread instead, which is
    # exactly what the inherited `aexecute` does for `execute`.

    async def als(self, path: str) -> LsResult:
        """Async version of `ls`, delegating to it in a worker thread."""
        return await asyncio.to_thread(self.ls, path)

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        """Async version of `read`, delegating to it in a worker thread."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async version of `write`, delegating to it in a worker thread."""
        return await asyncio.to_thread(self.write, file_path, content)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Async version of `edit`, delegating to it in a worker thread."""
        return await asyncio.to_thread(
            self.edit, file_path, old_string, new_string, replace_all
        )

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Async version of `glob`, delegating to it in a worker thread."""
        return await asyncio.to_thread(self.glob, pattern, path)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Async version of `grep`, with the base class's timeout guard."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.grep, pattern, path, glob, max_count=max_count),
                timeout=ASYNC_GREP_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "agrep timed out after %ds (pattern=%r, path=%r, glob=%r)",
                ASYNC_GREP_TIMEOUT,
                pattern,
                path,
                glob,
            )
            return GrepResult(
                error=(
                    f"Error: grep timed out after {ASYNC_GREP_TIMEOUT}s. Try a "
                    "more specific pattern or a narrower path."
                )
            )

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """Upload files to the session's flat ``/mnt/data`` store.

        Only ``/mnt/data/<name>`` is storable; any other path is rejected with
        ``invalid_path`` rather than flattened into the root under a name the
        caller never asked for.

        Args:
            files: List of ``(path, content)`` tuples.

        Returns:
            One ``FileUploadResponse`` per input file.  Errors are captured
            per-file rather than raised.
        """
        responses: list[FileUploadResponse] = []
        api_url = self._build_url("files", params={"path": _FILE_ROOT})
        logger.debug("upload_files: %d file(s) to %s", len(files), api_url)

        for path, content in files:
            filename = _session_file_name(path)
            if filename is None:
                responses.append(FileUploadResponse(path=path, error=INVALID_PATH))
                continue
            try:
                logger.debug(
                    "upload_files: uploading %s (%d bytes)", path, len(content)
                )
                multipart = [
                    ("file", (filename, BytesIO(content), "application/octet-stream"))
                ]
                resp = requests.post(
                    api_url,
                    # Per file, not per batch: a large batch can outlive a
                    # token captured once at the top.
                    headers=self._auth_headers(),
                    data={},
                    files=multipart,
                    timeout=_request_timeout(self._timeout),
                )
                resp.raise_for_status()
                logger.debug("upload_files: %s succeeded (%s)", path, resp.status_code)
                responses.append(FileUploadResponse(path=path))
            except requests.RequestException as e:
                logger.warning("upload_files: %s failed", path, exc_info=True)
                responses.append(
                    FileUploadResponse(path=path, error=_transfer_error(e, upload=True))
                )
        return responses

    def download_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        """Download files from the session's flat ``/mnt/data`` store.

        Mirrors `upload_files`: only ``/mnt/data/<name>`` is addressable.

        Args:
            paths: List of file paths to download.

        Returns:
            One ``FileDownloadResponse`` per input path.  Errors are captured
            per-file rather than raised.
        """
        responses: list[FileDownloadResponse] = []
        logger.debug("download_files: %d file(s)", len(paths))

        for path in paths:
            filename = _session_file_name(path)
            if filename is None:
                responses.append(FileDownloadResponse(path=path, error=INVALID_PATH))
                continue
            try:
                encoded = urllib.parse.quote(filename)
                api_url = self._build_url(f"files/{encoded}/content")
                logger.debug("download_files: GET %s", api_url)
                resp = requests.get(
                    api_url,
                    # Per file, matching `upload_files`.
                    headers=self._auth_headers(),
                    timeout=_request_timeout(self._timeout),
                )
                resp.raise_for_status()
                logger.debug(
                    "download_files: %s succeeded (%d bytes)",
                    path,
                    len(resp.content),
                )
                responses.append(FileDownloadResponse(path=path, content=resp.content))
            except requests.RequestException as e:
                logger.warning("download_files: %s failed", path, exc_info=True)
                responses.append(
                    FileDownloadResponse(
                        path=path, error=_transfer_error(e, upload=False)
                    )
                )
        return responses
