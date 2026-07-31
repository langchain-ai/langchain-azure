"""Azure Blob Storage backend for LangChain Deep Agents."""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from typing import Any, Callable, Optional, Union

import azure.core.credentials
import azure.core.credentials_async
import azure.identity
import azure.identity.aio
import wcmatch.glob as wcglob
from azure.core import MatchConditions
from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    HttpResponseError,
    ResourceModifiedError,
    ResourceNotFoundError,
)
from azure.storage.blob import BlobPrefix, ContainerClient
from azure.storage.blob.aio import BlobPrefix as AsyncBlobPrefix
from azure.storage.blob.aio import ContainerClient as AsyncContainerClient
from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    PERMISSION_DENIED,
    BackendProtocol,
    DeleteResult,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    perform_string_replacement,
    slice_read_response,
    validate_path,
)
from langchain_core._api import beta
from langchain_core._api.beta_decorator import warn_beta

from langchain_azure_storage._user_agent import USER_AGENT
from langchain_azure_storage.deepagents._utils import (
    build_file_info,
    from_blob_key,
    get_prefix_for_path,
    is_text_file,
    to_blob_key,
)

logger = logging.getLogger(__name__)

_BETA_MESSAGE = (
    "`AzureBlobBackend` is in public preview. "
    "Its API is not stable and may change in future versions."
)

_GLOB_FLAGS = wcglob.BRACE | wcglob.GLOBSTAR


class _InvalidGlob(Exception):
    """Raised by `_compile_glob` when a pattern cannot be compiled."""


def _compile_glob(pattern: str) -> Callable[[str], bool]:
    """Compile *pattern* into a matcher over search-root-relative paths.

    One shared contract for ``glob()`` and ``grep()``'s ``glob`` filter, so the
    same pattern selects the same files in both:

    * No ``/`` in the pattern: match the basename at any depth
      (``*.py`` matches ``src/app/main.py``).
    * Pattern contains ``/``: match the path relative to the search root, with
      ``**`` support (``src/**/*.py`` matches ``src/app/main.py``).
    * Leading ``/``: anchor to the search root, narrowing rather than widening
      (``/*.py`` matches ``top.py`` but not ``src/main.py``).

    Dotfiles are excluded from a bare ``*`` (no ``DOTMATCH``); match them
    explicitly with ``.*``.

    Raises:
        _InvalidGlob: If the pattern is malformed or expands past ``wcmatch``'s
            brace-expansion limit.
    """
    anchored = "/" in pattern
    try:
        compiled = wcglob.compile(pattern.lstrip("/"), flags=_GLOB_FLAGS)
    except Exception as exc:  # noqa: BLE001 - surfaced as a result error
        raise _InvalidGlob(f"Error: invalid glob pattern '{pattern}': {exc}") from exc

    if anchored:
        return lambda relative: bool(compiled.match(relative))
    return lambda relative: bool(compiled.match(PurePosixPath(relative).name))


# Credential types accepted by the backend.
_SDK_CREDENTIAL_TYPE = Optional[
    Union[
        azure.core.credentials.AzureSasCredential,
        azure.core.credentials.TokenCredential,
        azure.core.credentials_async.AsyncTokenCredential,
    ]
]


def _relative_path(virtual_path: str, base_path: str) -> str | None:
    """Path of *virtual_path* relative to *base_path*, or None if outside it."""
    if base_path == "/":
        return virtual_path[1:]

    prefix_with_slash = base_path + "/"
    if virtual_path.startswith(prefix_with_slash):
        return virtual_path[len(prefix_with_slash) :]
    if virtual_path == base_path:
        return virtual_path.split("/")[-1]
    return None


def _modified_at(blob: Any) -> str:
    """ISO 8601 last-modified timestamp of a listed blob, or ``""`` if unknown."""
    last_modified = getattr(blob, "last_modified", None)
    return last_modified.isoformat() if last_modified else ""


def _ls_result(items: list[Any], prefix: str) -> LsResult:
    """Build an ``LsResult`` from a delimited walk of a directory.

    ``items`` are the immediate children returned by ``walk_blobs`` -- either
    ``BlobPrefix`` for synthesized subdirectories or ``BlobProperties`` for
    files. The ``/`` delimiter makes the walk non-recursive, so subdirectories
    arrive pre-collapsed and no post-filtering is needed.
    """
    infos: list[FileInfo] = []
    for item in items:
        if isinstance(item, (BlobPrefix, AsyncBlobPrefix)):
            infos.append(
                build_file_info(path=from_blob_key(prefix, item.name), is_dir=True)
            )
            continue
        if item.name.endswith("/"):
            continue  # Skip pseudo-directory marker blobs.
        infos.append(
            build_file_info(
                path=from_blob_key(prefix, item.name),
                is_dir=False,
                size=item.size or 0,
                modified_at=_modified_at(item),
            )
        )

    infos.sort(key=lambda x: x.get("path", ""))
    return LsResult(entries=infos)


def _glob_result(
    blobs: list[Any],
    prefix: str,
    normalized_path: str,
    matcher: Callable[[str], bool],
) -> GlobResult:
    """Build a ``GlobResult`` from blobs accepted by *matcher*.

    *matcher* comes from `_compile_glob`, so ``glob()`` and ``grep()``'s
    ``glob`` filter share one contract: a slash-less pattern matches the
    basename at any depth, a pattern with ``/`` matches the search-root-relative
    path, and a leading ``/`` anchors to the root.
    """
    infos: list[FileInfo] = []
    for blob in blobs:
        virtual = from_blob_key(prefix, blob.name)
        relative = _relative_path(virtual, normalized_path)
        if relative is None:
            continue
        if matcher(relative):
            infos.append(
                build_file_info(
                    path=virtual,
                    is_dir=False,
                    size=blob.size or 0,
                    modified_at=_modified_at(blob),
                )
            )
    return GlobResult(matches=infos)


def _read_result_from_bytes(
    raw: Any, offset: int, limit: int, *, is_text: bool = True
) -> ReadResult:
    """Build a ``ReadResult`` from raw blob bytes.

    Returns raw (unformatted) content -- the Deep Agents middleware applies
    line numbering, the empty-file reminder, and base64 multimodal handling at
    the tool boundary. ``slice_read_response`` supplies the pagination metadata
    and clamps degenerate windows.

    Encoding is chosen the way the filesystem-like reference backends do it:
    files whose extension classifies as non-text (image/audio/video/file) are
    returned base64 without a decode attempt, so binary content that happens to
    be UTF-8 decodable is not mistaken for text. Text-classified blobs are
    decoded as UTF-8, falling back to base64 when the bytes are not valid UTF-8.
    """
    content_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
    if is_text:
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            return slice_read_response(
                {"content": text, "encoding": "utf-8"}, offset, limit
            )

    b64 = base64.b64encode(content_bytes).decode("ascii")
    return ReadResult(file_data={"content": b64, "encoding": "base64"})


def _grep_lines(content: str, pattern: str, virtual: str) -> list[GrepMatch]:
    """Return literal-substring matches for *pattern* within *content*."""
    matches: list[GrepMatch] = []
    for line_num, line in enumerate(content.split("\n"), 1):
        if pattern in line:
            matches.append({"path": virtual, "line": line_num, "text": line})
    return matches


def _grep_candidates(
    blobs: list[Any],
    prefix: str,
    search_path: str,
    matcher: Callable[[str], bool] | None,
) -> list[Any]:
    """Filter listed blobs to those inside *search_path* accepted by *matcher*.

    *matcher* comes from `_compile_glob` (``None`` when no ``glob`` filter was
    given), so the include filter follows the same contract as ``glob()``.
    """
    candidates: list[Any] = []
    for blob in blobs:
        virtual = from_blob_key(prefix, blob.name)
        relative = _relative_path(virtual, search_path)
        if relative is None:
            continue
        if matcher is not None and not matcher(relative):
            continue
        candidates.append(blob)
    return candidates


def _keys_to_delete(
    blobs: list[Any], exact_key: str | None, descendant_prefix: str
) -> list[str]:
    """Select the blob keys a recursive delete of one path should remove.

    The server-side listing prefix also matches siblings sharing a name stem
    (``/src`` returns ``/src-backup``), so this filter is what enforces the
    subtree boundary. Do not widen it to a bare prefix match.

    Marker blobs (keys ending in ``/``) are deliberately kept, unlike in `ls`:
    they sit inside the subtree being removed.
    """
    return [
        blob.name
        for blob in blobs
        if (exact_key is not None and blob.name == exact_key)
        or blob.name.startswith(descendant_prefix)
    ]


def _delete_failure_error(
    file_path: str, failed_paths: list[str], attempted: int
) -> str:
    """Build the error for a recursive delete that could not remove everything.

    A wholly failed delete is reported distinctly from a partial one: retrying
    the former is safe, the latter has already changed the container.
    """
    failed_paths.sort()
    sample = ", ".join(failed_paths[:3])
    remainder = len(failed_paths) - min(len(failed_paths), 3)
    suffix = f", and {remainder} more" if remainder else ""
    if len(failed_paths) == attempted:
        summary = f"failed on all {attempted} file(s)"
    else:
        removed = attempted - len(failed_paths)
        summary = f"removed {removed} file(s) but failed on {len(failed_paths)}"
    return f"Error: delete of '{file_path}' {summary}: {sample}{suffix}"


def _normalize_max_count(max_count: int | None) -> int | None:
    """Clamp a caller-supplied grep cap.

    The ``grep`` tool schema sets no lower bound on ``max_count``, so a model
    can send a negative value, which would otherwise slice from the end and
    silently drop matches. Clamping to ``0`` matches ``grep_matches_from_files``.
    """
    if max_count is None:
        return None
    return max(max_count, 0)


def _windows(items: list[Any], size: int) -> Iterator[list[Any]]:
    """Yield *items* in consecutive chunks of at most *size*."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _grep_failure_result(failed_blobs: list[str]) -> GrepResult:
    """Build the error ``GrepResult`` for blobs that could not be read."""
    failed_blobs.sort()
    sample = ", ".join(failed_blobs[:3])
    remainder = len(failed_blobs) - min(len(failed_blobs), 3)
    suffix = f", and {remainder} more" if remainder else ""
    return GrepResult(
        error=(
            f"Error: grep could not read {len(failed_blobs)} file(s): {sample}{suffix}"
        )
    )


class _InvalidPath(Exception):
    """Raised by `_validate_path` with a formatted, user-facing message."""


def _validate_path(path: str) -> str:
    """Validate *path*, returning the normalized path.

    Raises:
        _InvalidPath: With a formatted ``"Error: Invalid path '...': ..."``
            message, for the caller to put in the appropriate Result's
            ``error`` field.
    """
    try:
        return validate_path(path)
    except ValueError as exc:
        raise _InvalidPath(f"Error: Invalid path '{path}': {exc}") from exc


def _operation_error(exc: Exception) -> str:
    """Map an upload/download failure to a per-file response error.

    Only authentication/authorization failures are reported with the
    standardized ``PERMISSION_DENIED`` code; anything else returns the
    exception's message so the caller has something actionable (full details
    also go to the log).
    """
    if isinstance(exc, ClientAuthenticationError):
        return PERMISSION_DENIED
    if isinstance(exc, HttpResponseError) and exc.status_code in (401, 403):
        return PERMISSION_DENIED
    return str(exc)


@beta(message=_BETA_MESSAGE)
class AzureBlobBackend(BackendProtocol):
    """Azure Blob Storage filesystem backend for Deep Agents.

    Implements ``BackendProtocol`` using Azure Blob Storage as the persistence
    layer. File content is stored in blob bodies (UTF-8 text, or raw bytes for
    binary uploads). Directories are synthesized on the fly from blob key
    prefixes (no directory marker blobs).

    Two operations destroy data already in the container, both reachable as
    agent tools: :meth:`write` replaces an existing file in full, and
    :meth:`delete` removes a path plus everything nested under it. Deleting
    ``"/"`` empties the configured ``prefix`` namespace -- or, with no
    ``prefix``, the entire container. Set a ``prefix`` and enable container
    soft-delete/versioning to bound the blast radius.

    The underlying Azure SDK clients are created lazily on first use and
    cached; call :meth:`close`/:meth:`aclose` (or use the backend as a
    context manager) to release them. Because the cached async client is
    bound to the event loop it was first used on, drive a given backend
    instance's async methods from a single event loop.
    """

    _MAX_CONCURRENCY = 8

    def __init__(
        self,
        account_url: str,
        container_name: str,
        *,
        prefix: str | None = None,
        credential: _SDK_CREDENTIAL_TYPE = None,
    ) -> None:
        """Create a new backend instance authenticating via account URL + credential.

        Use :meth:`from_connection_string` instead to authenticate with a
        connection string (e.g. for the Azurite emulator).

        Args:
            account_url: Account URL, e.g.
                ``https://<account>.blob.core.windows.net``.
            container_name: Target blob container name.
            prefix: Optional key namespace prefix within the container. Scoping
                each agent/session to a prefix isolates their files.
            credential: Credential to authenticate with. If ``None``,
                ``DefaultAzureCredential`` is used.
        """
        self._account_url = account_url
        self._container_name = container_name
        self._prefix = prefix or ""
        self._credential = credential
        self._connection_string: str | None = None
        self._init_resource_state()

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        container_name: str,
        *,
        prefix: str | None = None,
    ) -> "AzureBlobBackend":
        """Create a new backend instance authenticating via a connection string.

        Intended for the `Azurite <https://learn.microsoft.com/azure/storage/common/storage-use-azurite>`_
        emulator, or any account where a connection string (rather than
        ``account_url`` + ``credential``) is more convenient.

        Args:
            connection_string: Full connection string (e.g. from the Azure
                portal, or for the Azurite emulator).
            container_name: Target blob container name.
            prefix: Optional key namespace prefix within the container.

        Returns:
            A new ``AzureBlobBackend`` authenticating via *connection_string*.
        """
        # The @beta wrapper on __init__ suppresses its warning for callers
        # inside langchain* packages (see langchain_core's is_caller_internal),
        # which includes this classmethod -- so emit it explicitly here.
        warn_beta(message=_BETA_MESSAGE)
        backend = cls("", container_name, prefix=prefix)
        backend._connection_string = connection_string
        return backend

    def _init_resource_state(self) -> None:
        self._sync_container_client: ContainerClient | None = None
        self._async_container_client: AsyncContainerClient | None = None
        self._sync_owned_credential: Any | None = None
        self._async_owned_credential: Any | None = None
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    # These validate that the provided credential matches the sync/async
    # method being used, following the same logic as AzureBlobStorageLoader's
    # in document_loaders.py. The shapes have since diverged: this backend
    # caches the container client and any credential it creates for reuse
    # across calls (see `_get_sync_container`/`_get_async_container`), rather
    # than creating one per call, so consolidating them isn't a pure move.
    def _resolve_sync_credential(
        self, provided_credential: _SDK_CREDENTIAL_TYPE
    ) -> _SDK_CREDENTIAL_TYPE:
        if provided_credential is None:
            credential = azure.identity.DefaultAzureCredential()
            self._sync_owned_credential = credential
            return credential
        if isinstance(
            provided_credential, azure.core.credentials_async.AsyncTokenCredential
        ):
            raise ValueError(
                "Cannot use synchronous methods when AzureBlobBackend is "
                "instantiated with an AsyncTokenCredential. Use the async "
                "methods instead, or supply a synchronous credential."
            )
        return provided_credential

    async def _resolve_async_credential(
        self, provided_credential: _SDK_CREDENTIAL_TYPE
    ) -> _SDK_CREDENTIAL_TYPE:
        if provided_credential is None:
            credential = azure.identity.aio.DefaultAzureCredential()
            self._async_owned_credential = credential
            return credential
        if not isinstance(
            provided_credential,
            (
                azure.core.credentials_async.AsyncTokenCredential,
                azure.core.credentials.AzureSasCredential,
            ),
        ):
            raise ValueError(
                "Cannot use asynchronous methods when AzureBlobBackend is "
                "instantiated with a synchronous TokenCredential. Use the sync "
                "methods instead, or supply an AsyncTokenCredential."
            )
        return provided_credential

    def _client_kwargs(self, credential: Any) -> dict[str, Any]:
        return {
            "account_url": self._account_url,
            "container_name": self._container_name,
            "credential": credential,
            "user_agent": USER_AGENT,
        }

    def _get_sync_container(self) -> ContainerClient:
        """Return the cached sync container client, creating it on first use.

        The client (and any credential this backend creates) is reused across
        calls -- creating a client/credential is expensive, and reuse gets us
        HTTP connection pooling and credential token caching. Call `close()`
        to release it.
        """
        if self._sync_container_client is not None:
            return self._sync_container_client
        with self._sync_lock:
            if self._sync_container_client is not None:
                return self._sync_container_client
            if self._connection_string:
                client = ContainerClient.from_connection_string(
                    self._connection_string,
                    self._container_name,
                    user_agent=USER_AGENT,
                )
            else:
                credential = self._resolve_sync_credential(self._credential)
                client = ContainerClient(**self._client_kwargs(credential))
            self._sync_container_client = client
            return client

    async def _get_async_container(self) -> AsyncContainerClient:
        """Return the cached async container client, creating it on first use.

        See `_get_sync_container` for why the client is cached. Call
        `aclose()` to release it.
        """
        if self._async_container_client is not None:
            return self._async_container_client
        async with self._async_lock:
            if self._async_container_client is not None:
                return self._async_container_client
            if self._connection_string:
                client = AsyncContainerClient.from_connection_string(
                    self._connection_string,
                    self._container_name,
                    user_agent=USER_AGENT,
                )
            else:
                credential = await self._resolve_async_credential(self._credential)
                client = AsyncContainerClient(**self._client_kwargs(credential))
            self._async_container_client = client
            return client

    def close(self) -> None:
        """Close the cached sync container client and any credential it owns.

        Only needed if any synchronous methods (``read``, ``write``, etc.)
        were called. A caller-supplied ``credential`` is caller-owned and is
        never closed here. Safe to call multiple times.
        """
        if self._sync_container_client is not None:
            self._sync_container_client.close()
            self._sync_container_client = None
        if self._sync_owned_credential is not None:
            self._sync_owned_credential.close()
            self._sync_owned_credential = None

    async def aclose(self) -> None:
        """Close the cached async container client and any credential it owns.

        Only needed if any asynchronous methods (``aread``, ``awrite``, etc.)
        were called. A caller-supplied ``credential`` is caller-owned and is
        never closed here. Safe to call multiple times.
        """
        if self._async_container_client is not None:
            await self._async_container_client.close()
            self._async_container_client = None
        if self._async_owned_credential is not None:
            await self._async_owned_credential.close()
            self._async_owned_credential = None

    def __enter__(self) -> "AzureBlobBackend":
        """Enter the context manager, returning this backend."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Exit the context manager, calling `close()`."""
        self.close()

    async def __aenter__(self) -> "AzureBlobBackend":
        """Enter the async context manager, returning this backend."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the async context manager, calling `aclose()`."""
        await self.aclose()

    def ls(self, path: str) -> LsResult:
        """List files and synthesized subdirectories at a path.

        Subdirectories are synthesized from blob key prefixes; no directory
        marker blobs are required.

        Args:
            path: Virtual directory path (e.g. ``"/src"``).

        Returns:
            An ``LsResult`` whose ``entries`` holds the immediate children, or
            whose ``error`` is set when the path is invalid.
        """
        try:
            normalized_root = _validate_path(path or "/")
        except _InvalidPath as exc:
            return LsResult(error=str(exc))

        container = self._get_sync_container()
        items = [
            item
            for item in container.walk_blobs(
                name_starts_with=get_prefix_for_path(self._prefix, normalized_root)
                or None,
                delimiter="/",
            )
        ]
        return _ls_result(items, self._prefix)

    async def als(self, path: str) -> LsResult:
        """List files and synthesized subdirectories at a path.

        Subdirectories are synthesized from blob key prefixes; no directory
        marker blobs are required.

        Args:
            path: Virtual directory path (e.g. ``"/src"``).

        Returns:
            An ``LsResult`` whose ``entries`` holds the immediate children, or
            whose ``error`` is set when the path is invalid.
        """
        try:
            normalized_root = _validate_path(path or "/")
        except _InvalidPath as exc:
            return LsResult(error=str(exc))

        container = await self._get_async_container()
        items = [
            item
            async for item in container.walk_blobs(
                name_starts_with=get_prefix_for_path(self._prefix, normalized_root)
                or None,
                delimiter="/",
            )
        ]
        return _ls_result(items, self._prefix)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a file and return its raw content for the requested window.

        The content is returned unformatted: the Deep Agents middleware applies
        line numbering, the empty-file reminder, and base64 multimodal handling
        based on the ``encoding`` field. Blobs whose extension classifies as
        non-text (image/audio/video/file), or whose bytes are not valid UTF-8,
        are returned base64-encoded with ``encoding="base64"``.

        Args:
            file_path: Virtual path to the file.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return.

        Returns:
            A ``ReadResult`` with the file content, or an error if the path is
            invalid, the file is not found, or the offset is out of range.
        """
        try:
            file_path = _validate_path(file_path)
        except _InvalidPath as exc:
            return ReadResult(error=str(exc))

        container = self._get_sync_container()
        try:
            raw = (
                container.get_blob_client(self._blob_key(file_path))
                .download_blob()
                .readall()
            )
        except ResourceNotFoundError:
            return ReadResult(error=f"File '{file_path}' not found")
        return _read_result_from_bytes(
            raw, offset, limit, is_text=is_text_file(file_path)
        )

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        """Read a file and return its raw content for the requested window.

        The content is returned unformatted: the Deep Agents middleware applies
        line numbering, the empty-file reminder, and base64 multimodal handling
        based on the ``encoding`` field. Blobs whose extension classifies as
        non-text (image/audio/video/file), or whose bytes are not valid UTF-8,
        are returned base64-encoded with ``encoding="base64"``.

        Args:
            file_path: Virtual path to the file.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return.

        Returns:
            A ``ReadResult`` with the file content, or an error if the path is
            invalid, the file is not found, or the offset is out of range.
        """
        try:
            file_path = _validate_path(file_path)
        except _InvalidPath as exc:
            return ReadResult(error=str(exc))

        container = await self._get_async_container()
        try:
            stream = await container.get_blob_client(
                self._blob_key(file_path)
            ).download_blob()
            raw = await stream.readall()
        except ResourceNotFoundError:
            return ReadResult(error=f"File '{file_path}' not found")
        return _read_result_from_bytes(
            raw, offset, limit, is_text=is_text_file(file_path)
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write content to a file, creating it or replacing it if it exists.

        An existing file is replaced in full. Use ``edit``/``aedit`` instead
        when the existing content must be preserved.

        Args:
            file_path: Virtual path for the file.
            content: UTF-8 text content to write.

        Returns:
            A ``WriteResult`` with the path, or an error if the path is
            invalid.
        """
        try:
            file_path = _validate_path(file_path)
        except _InvalidPath as exc:
            return WriteResult(error=str(exc))

        container = self._get_sync_container()
        container.get_blob_client(self._blob_key(file_path)).upload_blob(
            content.encode("utf-8"),
            overwrite=True,
        )
        return WriteResult(path=file_path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Write content to a file, creating it or replacing it if it exists.

        An existing file is replaced in full. Use ``edit``/``aedit`` instead
        when the existing content must be preserved.

        Args:
            file_path: Virtual path for the file.
            content: UTF-8 text content to write.

        Returns:
            A ``WriteResult`` with the path, or an error if the path is
            invalid.
        """
        try:
            file_path = _validate_path(file_path)
        except _InvalidPath as exc:
            return WriteResult(error=str(exc))

        container = await self._get_async_container()
        await container.get_blob_client(self._blob_key(file_path)).upload_blob(
            content.encode("utf-8"),
            overwrite=True,
        )
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace text in an existing file.

        Args:
            file_path: Virtual path to the file.
            old_string: Exact substring to find.
            new_string: Replacement text.
            replace_all: If ``True``, replace every occurrence; otherwise
                require exactly one match.

        Returns:
            An ``EditResult`` with the path and occurrence count, or an error
            if the path is invalid, the file is missing, the match is not
            unique, or the file was modified concurrently during the edit.
        """
        try:
            file_path = _validate_path(file_path)
        except _InvalidPath as exc:
            return EditResult(error=str(exc))

        container = self._get_sync_container()
        blob = container.get_blob_client(self._blob_key(file_path))
        try:
            downloader = blob.download_blob(encoding="utf-8")
            etag = downloader.properties.etag
            content = str(downloader.readall())
        except ResourceNotFoundError:
            return EditResult(error=f"Error: File '{file_path}' not found")

        result = perform_string_replacement(
            content, old_string, new_string, replace_all
        )
        if isinstance(result, str):
            return EditResult(error=result)
        new_content, occurrences = result
        try:
            blob.upload_blob(
                new_content.encode("utf-8"),
                overwrite=True,
                etag=etag,
                match_condition=MatchConditions.IfNotModified,
            )
        except ResourceModifiedError:
            return EditResult(error=_concurrent_modification_error(file_path))
        return EditResult(path=file_path, occurrences=int(occurrences))

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace text in an existing file.

        Args:
            file_path: Virtual path to the file.
            old_string: Exact substring to find.
            new_string: Replacement text.
            replace_all: If ``True``, replace every occurrence; otherwise
                require exactly one match.

        Returns:
            An ``EditResult`` with the path and occurrence count, or an error
            if the path is invalid, the file is missing, the match is not
            unique, or the file was modified concurrently during the edit.
        """
        try:
            file_path = _validate_path(file_path)
        except _InvalidPath as exc:
            return EditResult(error=str(exc))

        container = await self._get_async_container()
        blob = container.get_blob_client(self._blob_key(file_path))
        try:
            stream = await blob.download_blob(encoding="utf-8")
            etag = stream.properties.etag
            content = str(await stream.readall())
        except ResourceNotFoundError:
            return EditResult(error=f"Error: File '{file_path}' not found")

        result = perform_string_replacement(
            content, old_string, new_string, replace_all
        )
        if isinstance(result, str):
            return EditResult(error=result)
        new_content, occurrences = result
        try:
            await blob.upload_blob(
                new_content.encode("utf-8"),
                overwrite=True,
                etag=etag,
                match_condition=MatchConditions.IfNotModified,
            )
        except ResourceModifiedError:
            return EditResult(error=_concurrent_modification_error(file_path))
        return EditResult(path=file_path, occurrences=int(occurrences))

    def delete(self, file_path: str) -> DeleteResult:
        """Delete a file, or a directory and everything nested under it.

        Deletion is recursive: it removes the blob at *file_path* plus every
        blob whose key starts with ``file_path + "/"``. Deleting ``"/"``
        therefore removes every blob in the configured ``prefix`` namespace --
        or, when no ``prefix`` is set, every blob in the container. Scope the
        backend with ``prefix`` and enable container soft-delete/versioning if
        an agent should not be able to do that.

        Args:
            file_path: Virtual path of the file or directory to delete.

        Returns:
            A ``DeleteResult`` with the path, or an error if the path is
            invalid, nothing exists at or under it, or a blob could not be
            deleted.
        """
        try:
            file_path = _validate_path(file_path)
        except _InvalidPath as exc:
            return DeleteResult(error=str(exc))

        container = self._get_sync_container()
        exact_key, descendant_prefix = self._delete_scope(file_path)
        blobs = self._list_blobs_sync(
            container, exact_key if exact_key is not None else descendant_prefix
        )
        keys = _keys_to_delete(blobs, exact_key, descendant_prefix)
        if not keys:
            return DeleteResult(error=f"Error: File '{file_path}' not found")

        def remove(key: str) -> str | None:
            try:
                container.get_blob_client(key).delete_blob()
            except ResourceNotFoundError:
                return None  # Raced with another deleter; treat as done.
            except AzureError as exc:
                logger.error("Failed to delete %s: %s", key, exc)
                return from_blob_key(self._prefix, key)
            return None

        with ThreadPoolExecutor(max_workers=self._MAX_CONCURRENCY) as executor:
            failed = [path for path in executor.map(remove, keys) if path is not None]

        if failed:
            return DeleteResult(
                error=_delete_failure_error(file_path, failed, len(keys))
            )
        return DeleteResult(path=file_path)

    async def adelete(self, file_path: str) -> DeleteResult:
        """Delete a file, or a directory and everything nested under it.

        Deletion is recursive: it removes the blob at *file_path* plus every
        blob whose key starts with ``file_path + "/"``. Deleting ``"/"``
        therefore removes every blob in the configured ``prefix`` namespace --
        or, when no ``prefix`` is set, every blob in the container. Scope the
        backend with ``prefix`` and enable container soft-delete/versioning if
        an agent should not be able to do that.

        Args:
            file_path: Virtual path of the file or directory to delete.

        Returns:
            A ``DeleteResult`` with the path, or an error if the path is
            invalid, nothing exists at or under it, or a blob could not be
            deleted.
        """
        try:
            file_path = _validate_path(file_path)
        except _InvalidPath as exc:
            return DeleteResult(error=str(exc))

        container = await self._get_async_container()
        exact_key, descendant_prefix = self._delete_scope(file_path)
        blobs = await self._list_blobs_async(
            container, exact_key if exact_key is not None else descendant_prefix
        )
        keys = _keys_to_delete(blobs, exact_key, descendant_prefix)
        if not keys:
            return DeleteResult(error=f"Error: File '{file_path}' not found")

        semaphore = asyncio.Semaphore(self._MAX_CONCURRENCY)

        async def remove(key: str) -> str | None:
            async with semaphore:
                try:
                    await container.get_blob_client(key).delete_blob()
                except ResourceNotFoundError:
                    return None  # Raced with another deleter; treat as done.
                except AzureError as exc:
                    logger.error("Failed to delete %s: %s", key, exc)
                    return from_blob_key(self._prefix, key)
            return None

        failed = [
            path
            for path in await asyncio.gather(*(remove(key) for key in keys))
            if path is not None
        ]

        if failed:
            return DeleteResult(
                error=_delete_failure_error(file_path, failed, len(keys))
            )
        return DeleteResult(path=file_path)

    def _delete_scope(self, normalized: str) -> tuple[str | None, str]:
        """Resolve *normalized* to the blob keys a recursive delete may touch.

        Returns ``(exact_key, descendant_prefix)``. ``exact_key`` is ``None``
        for the root, which names no blob of its own. ``descendant_prefix``
        must keep its trailing slash -- that is what stops a delete of
        ``/workspace`` from reaching ``/workspace-backup``.
        """
        descendant_prefix = get_prefix_for_path(self._prefix, normalized)
        if normalized == "/":
            return None, descendant_prefix
        return to_blob_key(self._prefix, normalized), descendant_prefix

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern.

        Supports ``**`` (globstar) and ``{a,b}`` brace expansion. Uses
        the same matching contract as ``grep()``'s ``glob`` filter: a pattern
        without ``/`` matches the basename at any depth (``*.py`` matches
        ``/src/app/main.py``), a pattern containing ``/`` matches the path
        relative to *path* (``src/**/*.py``), and a leading ``/`` anchors to
        *path* (``/*.py`` matches only its immediate children). A bare ``*``
        does not match dotfiles; use ``.*`` for those.

        Args:
            pattern: Glob pattern (e.g. ``"*.py"``, ``"src/**/*.py"``).
            path: Base directory for the search (default: ``"/"``).

        Returns:
            A ``GlobResult`` whose ``matches`` holds the matching files, or
            whose ``error`` is set when the path or pattern is invalid.
        """
        try:
            normalized_path = _validate_path(path or "/")
            matcher = _compile_glob(pattern)
        except (_InvalidPath, _InvalidGlob) as exc:
            return GlobResult(error=str(exc))

        container = self._get_sync_container()
        blobs = self._list_target_blobs_sync(container, normalized_path)
        return _glob_result(blobs, self._prefix, normalized_path, matcher)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern.

        Supports ``**`` (globstar) and ``{a,b}`` brace expansion. Uses
        the same matching contract as ``grep()``'s ``glob`` filter: a pattern
        without ``/`` matches the basename at any depth (``*.py`` matches
        ``/src/app/main.py``), a pattern containing ``/`` matches the path
        relative to *path* (``src/**/*.py``), and a leading ``/`` anchors to
        *path* (``/*.py`` matches only its immediate children). A bare ``*``
        does not match dotfiles; use ``.*`` for those.

        Args:
            pattern: Glob pattern (e.g. ``"*.py"``, ``"src/**/*.py"``).
            path: Base directory for the search (default: ``"/"``).

        Returns:
            A ``GlobResult`` whose ``matches`` holds the matching files, or
            whose ``error`` is set when the path or pattern is invalid.
        """
        try:
            normalized_path = _validate_path(path or "/")
            matcher = _compile_glob(pattern)
        except (_InvalidPath, _InvalidGlob) as exc:
            return GlobResult(error=str(exc))

        container = await self._get_async_container()
        blobs = await self._list_target_blobs_async(container, normalized_path)
        return _glob_result(blobs, self._prefix, normalized_path, matcher)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search file contents for a literal substring.

        Args:
            pattern: Literal substring to search for.
            path: Directory scope for the search (default: ``"/"``).
            glob: Optional glob to pre-filter files, using the same contract as
                ``glob()``: a pattern without a slash (e.g. ``"*.py"``) matches
                file names at any depth, a pattern with a slash is matched
                against the path relative to *path*, and a leading ``/``
                anchors to *path*.
            max_count: Optional total cap on the number of matches returned
                across all files. ``None`` (the default) returns every match.
                When set, scanning stops once the cap is exceeded and the
                result is flagged ``truncated=True``; exactly *max_count*
                matches with none dropped is reported complete.

        Returns:
            A ``GrepResult`` whose ``matches`` holds matching lines, or whose
            ``error`` is set when the path or glob is invalid, or a file cannot
            be read.
        """
        try:
            search_path = _validate_path(path or "/")
            glob_matcher = _compile_glob(glob) if glob else None
        except (_InvalidPath, _InvalidGlob) as exc:
            return GrepResult(error=str(exc))

        failed: list[str] = []
        container = self._get_sync_container()
        blobs = self._list_target_blobs_sync(container, search_path)
        candidates = _grep_candidates(blobs, self._prefix, search_path, glob_matcher)

        def scan(blob: Any) -> list[GrepMatch]:
            virtual = from_blob_key(self._prefix, blob.name)
            try:
                content = str(
                    container.get_blob_client(blob.name)
                    .download_blob(encoding="utf-8")
                    .readall()
                )
            except (AzureError, UnicodeError) as exc:
                logger.warning("Failed to read blob %s for grep: %s", blob.name, exc)
                failed.append(virtual)
                return []
            return _grep_lines(content, pattern, virtual)

        cap = _normalize_max_count(max_count)
        # Only a capped scan needs to stop early, so only it pays for chunking.
        window = self._MAX_CONCURRENCY if cap is not None else max(len(candidates), 1)
        matches: list[GrepMatch] = []
        truncated = False
        with ThreadPoolExecutor(max_workers=self._MAX_CONCURRENCY) as executor:
            for chunk in _windows(candidates, window):
                for blob_matches in executor.map(scan, chunk):
                    matches.extend(blob_matches)
                if cap is not None and len(matches) > cap:
                    del matches[cap:]
                    truncated = True
                    break

        if failed:
            return _grep_failure_result(failed)
        return GrepResult(matches=matches, truncated=truncated)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search file contents for a literal substring.

        Args:
            pattern: Literal substring to search for.
            path: Directory scope for the search (default: ``"/"``).
            glob: Optional glob to pre-filter files, using the same contract as
                ``glob()``: a pattern without a slash (e.g. ``"*.py"``) matches
                file names at any depth, a pattern with a slash is matched
                against the path relative to *path*, and a leading ``/``
                anchors to *path*.
            max_count: Optional total cap on the number of matches returned
                across all files. ``None`` (the default) returns every match.
                When set, scanning stops once the cap is exceeded and the
                result is flagged ``truncated=True``; exactly *max_count*
                matches with none dropped is reported complete.

        Returns:
            A ``GrepResult`` whose ``matches`` holds matching lines, or whose
            ``error`` is set when the path or glob is invalid, or a file cannot
            be read.
        """
        try:
            search_path = _validate_path(path or "/")
            glob_matcher = _compile_glob(glob) if glob else None
        except (_InvalidPath, _InvalidGlob) as exc:
            return GrepResult(error=str(exc))

        failed: list[str] = []
        container = await self._get_async_container()
        blobs = await self._list_target_blobs_async(container, search_path)
        candidates = _grep_candidates(blobs, self._prefix, search_path, glob_matcher)

        # Concurrency bound must not come from the chunking below: an uncapped
        # scan is a single chunk, so removing this opens unbounded connections.
        semaphore = asyncio.Semaphore(self._MAX_CONCURRENCY)

        async def scan(blob: Any) -> list[GrepMatch]:
            virtual = from_blob_key(self._prefix, blob.name)
            async with semaphore:
                try:
                    stream = await container.get_blob_client(blob.name).download_blob(
                        encoding="utf-8"
                    )
                    content = str(await stream.readall())
                except (AzureError, UnicodeError) as exc:
                    logger.warning(
                        "Failed to read blob %s for grep: %s", blob.name, exc
                    )
                    failed.append(virtual)
                    return []
            return _grep_lines(content, pattern, virtual)

        cap = _normalize_max_count(max_count)
        # Only a capped scan needs to stop early, so only it pays for chunking.
        window = self._MAX_CONCURRENCY if cap is not None else max(len(candidates), 1)
        matches: list[GrepMatch] = []
        truncated = False
        for chunk in _windows(candidates, window):
            for blob_matches in await asyncio.gather(*(scan(b) for b in chunk)):
                matches.extend(blob_matches)
            if cap is not None and len(matches) > cap:
                del matches[cap:]
                truncated = True
                break

        if failed:
            return _grep_failure_result(failed)
        return GrepResult(matches=matches, truncated=truncated)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload one or more binary files, overwriting any that exist.

        Args:
            files: List of ``(path, content_bytes)`` tuples.

        Returns:
            A list of ``FileUploadResponse`` objects, one per input file.
        """
        container = self._get_sync_container()

        def upload(file: tuple[str, bytes]) -> FileUploadResponse:
            file_path, content = file
            try:
                validated = validate_path(file_path)
            except ValueError:
                return FileUploadResponse(path=file_path, error=INVALID_PATH)
            try:
                container.get_blob_client(self._blob_key(validated)).upload_blob(
                    content, overwrite=True
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to upload %s: %s", validated, exc)
                return FileUploadResponse(path=validated, error=_operation_error(exc))
            return FileUploadResponse(path=validated, error=None)

        with ThreadPoolExecutor(max_workers=self._MAX_CONCURRENCY) as executor:
            return list(executor.map(upload, files))

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload one or more binary files, overwriting any that exist.

        Args:
            files: List of ``(path, content_bytes)`` tuples.

        Returns:
            A list of ``FileUploadResponse`` objects, one per input file.
        """
        container = await self._get_async_container()
        semaphore = asyncio.Semaphore(self._MAX_CONCURRENCY)

        async def upload(file_path: str, content: bytes) -> FileUploadResponse:
            try:
                validated = validate_path(file_path)
            except ValueError:
                return FileUploadResponse(path=file_path, error=INVALID_PATH)
            async with semaphore:
                try:
                    await container.get_blob_client(
                        self._blob_key(validated)
                    ).upload_blob(content, overwrite=True)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to upload %s: %s", validated, exc)
                    return FileUploadResponse(
                        path=validated, error=_operation_error(exc)
                    )
            return FileUploadResponse(path=validated, error=None)

        return list(
            await asyncio.gather(*(upload(path, content) for path, content in files))
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download one or more files as raw bytes.

        Args:
            paths: Virtual paths to download.

        Returns:
            A list of ``FileDownloadResponse`` objects, one per input path;
            each has ``content`` on success or an error code on failure.
        """
        container = self._get_sync_container()

        def download(file_path: str) -> FileDownloadResponse:
            try:
                validated = validate_path(file_path)
            except ValueError:
                return FileDownloadResponse(
                    path=file_path, content=None, error=INVALID_PATH
                )
            try:
                raw = (
                    container.get_blob_client(self._blob_key(validated))
                    .download_blob()
                    .readall()
                )
            except ResourceNotFoundError:
                return FileDownloadResponse(
                    path=validated, content=None, error=FILE_NOT_FOUND
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to download %s: %s", validated, exc)
                return FileDownloadResponse(
                    path=validated, content=None, error=_operation_error(exc)
                )
            return FileDownloadResponse(
                path=validated, content=_as_bytes(raw), error=None
            )

        with ThreadPoolExecutor(max_workers=self._MAX_CONCURRENCY) as executor:
            return list(executor.map(download, paths))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download one or more files as raw bytes.

        Args:
            paths: Virtual paths to download.

        Returns:
            A list of ``FileDownloadResponse`` objects, one per input path;
            each has ``content`` on success or an error code on failure.
        """
        container = await self._get_async_container()
        semaphore = asyncio.Semaphore(self._MAX_CONCURRENCY)

        async def download(file_path: str) -> FileDownloadResponse:
            try:
                validated = validate_path(file_path)
            except ValueError:
                return FileDownloadResponse(
                    path=file_path, content=None, error=INVALID_PATH
                )
            async with semaphore:
                try:
                    stream = await container.get_blob_client(
                        self._blob_key(validated)
                    ).download_blob()
                    raw = await stream.readall()
                except ResourceNotFoundError:
                    return FileDownloadResponse(
                        path=validated, content=None, error=FILE_NOT_FOUND
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to download %s: %s", validated, exc)
                    return FileDownloadResponse(
                        path=validated, content=None, error=_operation_error(exc)
                    )
            return FileDownloadResponse(
                path=validated, content=_as_bytes(raw), error=None
            )

        return list(await asyncio.gather(*(download(path) for path in paths)))

    def _blob_key(self, path: str) -> str:
        return to_blob_key(self._prefix, path)

    def _list_target_blobs_sync(
        self, container: ContainerClient, path: str
    ) -> list[Any]:
        # glob()/grep() treat *path* as a directory (per the BackendProtocol
        # contract), so we recursively list everything under its prefix.
        return self._list_blobs_sync(container, get_prefix_for_path(self._prefix, path))

    async def _list_target_blobs_async(
        self, container: AsyncContainerClient, path: str
    ) -> list[Any]:
        # glob()/grep() treat *path* as a directory (per the BackendProtocol
        # contract), so we recursively list everything under its prefix.
        return await self._list_blobs_async(
            container, get_prefix_for_path(self._prefix, path)
        )

    def _list_blobs_sync(self, container: ContainerClient, prefix: str) -> list[Any]:
        return [blob for blob in container.list_blobs(name_starts_with=prefix or None)]

    async def _list_blobs_async(
        self, container: AsyncContainerClient, prefix: str
    ) -> list[Any]:
        return [
            blob async for blob in container.list_blobs(name_starts_with=prefix or None)
        ]


def _concurrent_modification_error(file_path: str) -> str:
    return (
        f"Error: File '{file_path}' was modified concurrently during the "
        f"edit. Read the file again and retry the edit."
    )


def _as_bytes(raw: Any) -> bytes:
    return raw if isinstance(raw, bytes) else raw.encode("utf-8")
