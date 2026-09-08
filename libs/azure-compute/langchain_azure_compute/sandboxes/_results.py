"""Pure conversions between the ACA SDK and Deep Agents result types.

No I/O: everything here turns bytes, an `ExecResult`, or an exception into the
protocol type `ACASandbox` returns, so it can be tested without a client.
"""

from __future__ import annotations

import base64
import logging
import shlex
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError
from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileData,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import (
    MAX_BINARY_BYTES,
    MAX_OUTPUT_BYTES,
    TRUNCATION_MSG,
)
from deepagents.backends.utils import normalize_read_bounds

if TYPE_CHECKING:
    from azure.containerapps.sandbox import ExecResult
    from azure.core.exceptions import AzureError

logger = logging.getLogger(__name__)

# GNU coreutils `timeout` exit status when the command was killed on expiry.
_TIMEOUT_EXIT_CODE = 124


# Vendored from `deepagents.backends.utils` (0.7.1): the extensions the
# first-party backends route through the base64 read path, plus the extra video
# containers `_get_backend_read_file_type` forces to binary.
#
# deepagents exposes no public equivalent -- `_EXTENSION_TO_FILE_TYPE`,
# `_VIDEO_EXTRA_EXTENSIONS`, `_get_file_type` and `_get_backend_read_file_type`
# are all private, so importing one would break on any upstream rename. This
# copy is kept honest by a parity test in
# `tests/unit_tests/sandboxes/sandbox/test_results.py`, which fails if it
# diverges from the installed version. Switch to the upstream helper if one
# becomes public.
_NON_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Images
        ".png",
        ".jpeg",
        ".jpg",
        ".webp",
        ".gif",
        ".heic",
        ".heif",
        # Video
        ".mp4",
        ".mpeg",
        ".mov",
        ".avi",
        ".flv",
        ".mpg",
        ".webm",
        ".wmv",
        ".3gpp",
        ".mkv",
        # Audio
        ".wav",
        ".mp3",
        ".aiff",
        ".aac",
        ".ogg",
        ".flac",
        # Documents
        ".pdf",
        ".ppt",
        ".pptx",
    }
)


def is_text_file(file_path: str) -> bool:
    """Whether `file_path`'s extension reads as text rather than base64.

    Extensions absent from the non-text set -- including no extension at all --
    default to text, matching the Deep Agents reference backends.
    """
    return PurePosixPath(file_path).suffix.lower() not in _NON_TEXT_EXTENSIONS


def execute_response(
    result: ExecResult, *, timeout_wrapped: bool = False
) -> ExecuteResponse:
    """Build an `ExecuteResponse` from an ACA SDK `ExecResult`.

    The timed-out note is only appended when wrapping was *requested* for this
    command (`timeout_wrapped`): without that, a command that happens to exit
    124 on its own would be narrated to the model as a timeout. Residual
    ambiguity remains in two directions -- under the wrapper, a genuine inner
    exit 124 is indistinguishable from expiry, and on an image without
    coreutils `timeout` the wrapper silently degrades to running untimed --
    both inherent to `timeout`'s exit-status convention and the `command -v`
    fallback.
    """
    output = result.stdout or ""
    if result.stderr:
        output += "\n" + result.stderr if output else result.stderr
    if timeout_wrapped and result.exit_code == _TIMEOUT_EXIT_CODE:
        note = f"Command timed out (exit {_TIMEOUT_EXIT_CODE})."
        output += "\n" + note if output else note
    return ExecuteResponse(output=output, exit_code=result.exit_code, truncated=False)


def execute_error_response(exc: Exception) -> ExecuteResponse:
    """Build the `ExecuteResponse` for a request that failed before completion.

    Raising instead would hand the agent loop a transport exception for what
    is, from its point of view, a command that produced no result. The most
    common cause is a command outrunning the HTTP transport's read timeout
    (~300 s on the SDK default), so the message says how to fix exactly that.
    The command may still be running server-side.

    Classified rather than interpolated: azure-core embeds the response body
    and request URL in its messages, and this string is handed to the model.
    """
    return ExecuteResponse(
        output=(
            "Error: the sandbox request failed before the command completed "
            f"({transfer_error(exc, missing_is_file=False)}). The command may "
            "still be running. If it "
            "legitimately runs longer than the HTTP transport's read timeout "
            "(about 300 seconds by default), construct the SandboxClient with "
            "a transport whose read timeout exceeds the command timeout."
        ),
        exit_code=None,
        truncated=False,
    )


def with_timeout(command: str, timeout: int | None) -> str:
    """Wrap `command` so it is killed after `timeout` seconds.

    `SandboxClient.exec()` has no server-side command timeout, and azure-core's
    `timeout` kwarg bounds the HTTP request rather than the process, so the
    protocol's timeout is enforced by wrapping the command instead. The HTTP
    request itself is still bounded by the client transport's read timeout
    (~300 s on the SDK default): a `timeout` above that is only reachable with
    a client constructed on a longer-read-timeout transport, which
    `ACASandbox`'s docs call out.

    Quote `command`: it must reach `sh -c` as a single argument, or a command
    containing a quote could close it and disable the very timeout being
    applied. `command -v` guards images that ship no coreutils `timeout`
    (minimal or BYO OCI disk images), which would otherwise fail with 127
    instead of running. `sh` means POSIX-shell semantics (dash on Debian-based
    presets) for every wrapped command; bashisms that would work against a
    bash-served `exec` fail here, which is the wrapper's documented cost.
    No `-k` kill-after: a TERM-ignoring child outrunning TERM will next hit
    the transport timeout, at which point the request fails anyway.
    """
    if timeout is None or timeout <= 0:
        return command
    inner = shlex.quote(command)
    return (
        f"if command -v timeout >/dev/null 2>&1; "
        f"then timeout {int(timeout)} sh -c {inner}; "
        f"else sh -c {inner}; fi"
    )


def binary_too_large_error(file_path: str) -> ReadResult:
    """The oversized-binary refusal, shared with the pre-download stat check."""
    return ReadResult(
        error=(
            f"File '{file_path}': Binary file exceeds maximum preview size "
            f"of {MAX_BINARY_BYTES} bytes"
        )
    )


def _binary_read_result(file_path: str, raw: bytes) -> ReadResult:
    """Build the base64 `ReadResult` the base class produces for binary files."""
    if len(raw) > MAX_BINARY_BYTES:
        return binary_too_large_error(file_path)
    return ReadResult(
        file_data=FileData(
            content=base64.b64encode(raw).decode("ascii"), encoding="base64"
        )
    )


def read_result_from_bytes(  # noqa: PLR0911 - one return per distinct outcome
    file_path: str, raw: bytes, *, offset: int, limit: int
) -> ReadResult:
    r"""Turn raw file bytes into the `ReadResult` the base class would return.

    Mirrors `BaseSandbox`'s server-side read script: empty files surface the
    reminder, non-text files (by extension, or that fail to decode) come back
    base64-capped, and text is newline-normalized, paginated, and byte-capped.

    Universal-newline normalization matters: without it CRLF files round-trip
    with stray `\r`, which then breaks `edit()` on the returned content.
    """
    if not raw:
        return ReadResult(
            file_data=FileData(
                content="System reminder: File exists but has empty contents",
                encoding="utf-8",
            )
        )

    if not is_text_file(file_path):
        return _binary_read_result(file_path, raw)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.info(
            "Text-extension file %s contained invalid UTF-8; returning as base64",
            file_path,
        )
        return _binary_read_result(file_path, raw)

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    offset, limit = normalize_read_bounds(offset, limit)
    if limit <= 0:
        return ReadResult(
            file_data=FileData(content="", encoding="utf-8"), no_lines_requested=True
        )

    total_lines = len(lines)
    if not lines or offset >= total_lines:
        return ReadResult(
            error=(
                f"File '{file_path}': Line offset {offset} exceeds file length "
                f"({total_lines} lines)"
            )
        )

    page = lines[offset : offset + limit]
    content = "\n".join(page)
    returned_lines = len(page)

    encoded = content.encode("utf-8")
    effective_limit = MAX_OUTPUT_BYTES - len(TRUNCATION_MSG.encode("utf-8"))
    if len(encoded) > effective_limit:
        truncated = encoded[:effective_limit].decode("utf-8", errors="ignore")
        # Advance only past fully rendered lines (each followed by "\n") so a
        # re-read from next_offset cannot skip an unshown line; the partial
        # boundary line is re-read from its start. Fall back to 1 so a first
        # line larger than the cap still makes forward progress.
        returned_lines = truncated.count("\n") or 1
        content = truncated + TRUNCATION_MSG

    end_line = offset + returned_lines
    return ReadResult(
        file_data=FileData(content=content, encoding="utf-8"),
        total_lines=total_lines,
        start_line=offset + 1,
        end_line=end_line,
        next_offset=end_line if end_line < total_lines else None,
    )


def transfer_error(exc: Exception, *, missing_is_file: bool = True) -> str:
    """Classify a sandbox failure for a result's `error`.

    Unrecognized failures get a bounded description rather than defaulting to a
    code, because a wrong code sends the agent down the wrong recovery path.

    `missing_is_file` decides what a 404 means, which depends on the operation
    rather than the exception. Reading or downloading a path that is not there
    is `file_not_found`; a 404 from `exec`, a write, or an upload is not -- the
    command has no file to miss, and an upload's destination is *expected* not
    to exist, so the 404 is the sandbox or its route. Reporting those as
    `file_not_found` sends the model into file recovery for an unavailable
    sandbox. The dynamic-sessions transfer classifier draws the same line.

    Do not interpolate the exception: azure-core embeds the response body in
    its messages, and this string is handed to the model.
    """
    if isinstance(exc, ResourceNotFoundError):
        if missing_is_file:
            return FILE_NOT_FOUND
        return "the sandbox or the requested route was not found (HTTP 404)"
    if isinstance(exc, ClientAuthenticationError):
        return PERMISSION_DENIED
    message = str(exc).lower()
    if "is a directory" in message or "isadirectory" in message:
        return IS_DIRECTORY
    # Typed checks and the status code outrank the *denied* sniff below: a
    # message that merely contains "denied" (a path name, a quoted server
    # detail) must not steer the agent into permission-recovery. The
    # directory sniff above deliberately runs first -- a genuine
    # directory-read failure arrives as an HTTP error whose body is the only
    # signal, so putting the status first would mask it.
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return PERMISSION_DENIED
    if status is not None:
        return f"request failed with HTTP {status}"
    if "permission" in message or "denied" in message:
        return PERMISSION_DENIED
    return f"request failed: {type(exc).__name__}"


def read_error(file_path: str, exc: AzureError) -> ReadResult:
    """Map a read failure to a `ReadResult`, shared by the sync/async forks."""
    if isinstance(exc, ResourceNotFoundError):
        return ReadResult(error=f"File '{file_path}': {FILE_NOT_FOUND}")
    # Classified rather than interpolated: azure-core embeds the response body
    # in its messages and this reaches the model. Full detail goes to the log.
    logger.warning("ACA sandbox read failed for %s: %s", file_path, exc)
    return ReadResult(error=f"File '{file_path}': {transfer_error(exc)}")


def write_error(file_path: str, exc: AzureError) -> WriteResult:
    """Map a write failure to a `WriteResult`, shared by the sync/async forks."""
    logger.warning("ACA sandbox write failed for %s: %s", file_path, exc)
    return WriteResult(
        error=(
            f"Failed to write file '{file_path}': "
            f"{transfer_error(exc, missing_is_file=False)}"
        )
    )
