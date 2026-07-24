"""Azure Blob Storage backend for LangChain Deep Agents."""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union

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
    ResourceExistsError,
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
    blobs: list[Any], prefix: str, normalized_path: str, pattern: str
) -> GlobResult:
    """Build a ``GlobResult`` by matching listed blobs against *pattern*.

    Uses shell-glob semantics (per the ``BackendProtocol`` docs): ``*`` stays
    within a single path segment and ``**`` is the recursion operator, so
    ``*.py`` matches only *path*'s immediate children and ``**/*.py`` matches
    at any depth. This differs from ``grep()``'s ``glob`` filter, which follows
    ripgrep ``--glob`` (a slash-less pattern matches the basename at any depth).
    """
    infos: list[FileInfo] = []
    for blob in blobs:
        virtual = from_blob_key(prefix, blob.name)
        relative = _relative_path(virtual, normalized_path)
        if relative is None:
            continue
        if wcglob.globmatch(relative, pattern, flags=_GLOB_FLAGS):
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
    the tool boundary.

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
            sliced = slice_read_response(
                {"content": text, "encoding": "utf-8"}, offset, limit
            )
            if isinstance(sliced, ReadResult):
                return sliced
            return ReadResult(file_data={"content": sliced, "encoding": "utf-8"})

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
    blobs: list[Any], prefix: str, search_path: str, glob: str | None
) -> list[Any]:
    """Filter listed blobs to those inside *search_path* matching *glob*.

    Mirroring ripgrep's ``--glob`` semantics, a pattern without a slash is
    matched against the file name at any depth; a pattern with a slash is
    matched against the path relative to *search_path*.
    """
    candidates: list[Any] = []
    for blob in blobs:
        virtual = from_blob_key(prefix, blob.name)
        relative = _relative_path(virtual, search_path)
        if relative is None:
            continue
        if glob:
            target = relative if "/" in glob else relative.rsplit("/", 1)[-1]
            if not wcglob.globmatch(target, glob, flags=_GLOB_FLAGS):
                continue
        candidates.append(blob)
    return candidates


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
        """Create a new file with the given content.

        Fails if the file already exists; use ``edit``/``aedit`` to modify an
        existing file.

        Args:
            file_path: Virtual path for the new file.
            content: UTF-8 text content to write.

        Returns:
            A ``WriteResult`` with the path, or an error if the path is
            invalid or the file exists.
        """
        try:
            file_path = _validate_path(file_path)
        except _InvalidPath as exc:
            return WriteResult(error=str(exc))

        container = self._get_sync_container()
        try:
            container.get_blob_client(self._blob_key(file_path)).upload_blob(
                content.encode("utf-8"),
                overwrite=False,
            )
        except ResourceExistsError:
            return WriteResult(error=_already_exists_error(file_path))
        return WriteResult(path=file_path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Create a new file with the given content.

        Fails if the file already exists; use ``edit``/``aedit`` to modify an
        existing file.

        Args:
            file_path: Virtual path for the new file.
            content: UTF-8 text content to write.

        Returns:
            A ``WriteResult`` with the path, or an error if the path is
            invalid or the file exists.
        """
        try:
            file_path = _validate_path(file_path)
        except _InvalidPath as exc:
            return WriteResult(error=str(exc))

        container = await self._get_async_container()
        try:
            await container.get_blob_client(self._blob_key(file_path)).upload_blob(
                content.encode("utf-8"),
                overwrite=False,
            )
        except ResourceExistsError:
            return WriteResult(error=_already_exists_error(file_path))
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

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern.

        Supports ``**`` (globstar) and ``{a,b}`` brace expansion. Uses
        shell-glob semantics: ``*.py`` matches only *path*'s immediate children;
        use ``**/*.py`` to match at any depth. (This differs from ``grep()``'s
        ``glob`` filter, which follows ripgrep and matches a slash-less pattern
        against the basename at any depth.)

        Args:
            pattern: Glob pattern (e.g. ``"**/*.py"``).
            path: Base directory for the search (default: ``"/"``).

        Returns:
            A ``GlobResult`` whose ``matches`` holds the matching files, or
            whose ``error`` is set when the path is invalid.
        """
        try:
            normalized_path = _validate_path(path or "/")
        except _InvalidPath as exc:
            return GlobResult(error=str(exc))

        container = self._get_sync_container()
        blobs = self._list_target_blobs_sync(container, normalized_path)
        return _glob_result(blobs, self._prefix, normalized_path, pattern)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern.

        Supports ``**`` (globstar) and ``{a,b}`` brace expansion. Uses
        shell-glob semantics: ``*.py`` matches only *path*'s immediate children;
        use ``**/*.py`` to match at any depth. (This differs from ``grep()``'s
        ``glob`` filter, which follows ripgrep and matches a slash-less pattern
        against the basename at any depth.)

        Args:
            pattern: Glob pattern (e.g. ``"**/*.py"``).
            path: Base directory for the search (default: ``"/"``).

        Returns:
            A ``GlobResult`` whose ``matches`` holds the matching files, or
            whose ``error`` is set when the path is invalid.
        """
        try:
            normalized_path = _validate_path(path or "/")
        except _InvalidPath as exc:
            return GlobResult(error=str(exc))

        container = await self._get_async_container()
        blobs = await self._list_target_blobs_async(container, normalized_path)
        return _glob_result(blobs, self._prefix, normalized_path, pattern)

    def grep(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> GrepResult:
        """Search file contents for a literal substring.

        Args:
            pattern: Literal substring to search for.
            path: Directory scope for the search (default: ``"/"``).
            glob: Optional glob to pre-filter files. Like ripgrep's
                ``--glob``, a pattern without a slash (e.g. ``"*.py"``)
                matches file names at any depth; a pattern with a slash is
                matched against the path relative to *path*.

        Returns:
            A ``GrepResult`` whose ``matches`` holds matching lines, or whose
            ``error`` is set when the path is invalid or a file cannot be read.
        """
        try:
            search_path = _validate_path(path or "/")
        except _InvalidPath as exc:
            return GrepResult(error=str(exc))

        matches: list[GrepMatch] = []
        failed: list[str] = []
        container = self._get_sync_container()
        blobs = self._list_target_blobs_sync(container, search_path)
        candidates = _grep_candidates(blobs, self._prefix, search_path, glob)

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

        with ThreadPoolExecutor(max_workers=self._MAX_CONCURRENCY) as executor:
            for blob_matches in executor.map(scan, candidates):
                matches.extend(blob_matches)

        if failed:
            return _grep_failure_result(failed)
        return GrepResult(matches=matches)

    async def agrep(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> GrepResult:
        """Search file contents for a literal substring.

        Args:
            pattern: Literal substring to search for.
            path: Directory scope for the search (default: ``"/"``).
            glob: Optional glob to pre-filter files. Like ripgrep's
                ``--glob``, a pattern without a slash (e.g. ``"*.py"``)
                matches file names at any depth; a pattern with a slash is
                matched against the path relative to *path*.

        Returns:
            A ``GrepResult`` whose ``matches`` holds matching lines, or whose
            ``error`` is set when the path is invalid or a file cannot be read.
        """
        try:
            search_path = _validate_path(path or "/")
        except _InvalidPath as exc:
            return GrepResult(error=str(exc))

        matches: list[GrepMatch] = []
        failed: list[str] = []
        container = await self._get_async_container()
        blobs = await self._list_target_blobs_async(container, search_path)
        candidates = _grep_candidates(blobs, self._prefix, search_path, glob)
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

        for blob_matches in await asyncio.gather(*(scan(b) for b in candidates)):
            matches.extend(blob_matches)

        if failed:
            return _grep_failure_result(failed)
        return GrepResult(matches=matches)

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


def _already_exists_error(file_path: str) -> str:
    return (
        f"Cannot write to {file_path} because it already exists. "
        f"Read and then make an edit, or write to a new path."
    )


def _concurrent_modification_error(file_path: str) -> str:
    return (
        f"Error: File '{file_path}' was modified concurrently during the "
        f"edit. Read the file again and retry the edit."
    )


def _as_bytes(raw: Any) -> bytes:
    return raw if isinstance(raw, bytes) else raw.encode("utf-8")
