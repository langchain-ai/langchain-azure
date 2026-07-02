"""Azure Blob Storage backend for LangChain Deep Agents."""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional, Union

import azure.core.credentials
import azure.core.credentials_async
import azure.identity
import azure.identity.aio
import wcmatch.glob as wcglob
from azure.core.exceptions import (
    AzureError,
    ResourceExistsError,
    ResourceNotFoundError,
)
from azure.storage.blob import ContainerClient
from azure.storage.blob.aio import ContainerClient as AsyncContainerClient
from deepagents.backends.protocol import (
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

from langchain_azure_storage._user_agent import USER_AGENT
from langchain_azure_storage.deepagents._path import (
    from_blob_key,
    get_prefix_for_path,
    to_blob_key,
)
from langchain_azure_storage.deepagents._utils import build_file_info

logger = logging.getLogger(__name__)

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


def _blob_view(name: str, size: int, metadata: dict[str, str] | None) -> Any:
    """Lightweight stand-in for a listed blob (used for exact-key matches)."""
    return SimpleNamespace(name=name, size=size, metadata=metadata)


def _ls_result(blobs: list[Any], prefix: str, normalized_root: str) -> LsResult:
    """Build an ``LsResult`` from listed blobs, synthesizing subdirectories."""
    infos: list[FileInfo] = []
    subdirs: set[str] = set()
    normalized_path = (
        normalized_root if normalized_root.endswith("/") else normalized_root + "/"
    )

    for blob in blobs:
        virtual = from_blob_key(prefix, blob.name)
        if not virtual.startswith(normalized_path):
            continue
        relative = virtual[len(normalized_path) :]
        if not relative:
            continue

        if "/" in relative:
            subdir_name = relative.split("/")[0]
            subdirs.add(normalized_path + subdir_name + "/")
        else:
            modified_at = blob.metadata.get("modified_at", "") if blob.metadata else ""
            infos.append(
                build_file_info(
                    path=virtual,
                    is_dir=False,
                    size=blob.size or 0,
                    modified_at=modified_at,
                )
            )

    for subdir in sorted(subdirs):
        infos.append(build_file_info(path=subdir, is_dir=True, size=0))

    infos.sort(key=lambda x: x.get("path", ""))
    return LsResult(entries=infos)


def _glob_result(
    blobs: list[Any], prefix: str, normalized_path: str, pattern: str
) -> GlobResult:
    """Build a ``GlobResult`` by matching listed blobs against *pattern*."""
    infos: list[FileInfo] = []
    for blob in blobs:
        virtual = from_blob_key(prefix, blob.name)
        relative = _relative_path(virtual, normalized_path)
        if relative is None:
            continue
        if wcglob.globmatch(relative, pattern, flags=_GLOB_FLAGS):
            modified_at = blob.metadata.get("modified_at", "") if blob.metadata else ""
            infos.append(
                build_file_info(
                    path=virtual,
                    is_dir=False,
                    size=blob.size or 0,
                    modified_at=modified_at,
                )
            )
    return GlobResult(matches=infos)


def _read_result_from_bytes(raw: Any, offset: int, limit: int) -> ReadResult:
    """Build a ``ReadResult`` from raw blob bytes.

    Returns raw (unformatted) content -- the Deep Agents middleware applies
    line numbering, the empty-file reminder, and base64 multimodal handling at
    the tool boundary. Blobs that are not valid UTF-8 are returned
    base64-encoded with ``encoding="base64"``.
    """
    content_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        b64 = base64.b64encode(content_bytes).decode("ascii")
        return ReadResult(file_data={"content": b64, "encoding": "base64"})

    sliced = slice_read_response({"content": text, "encoding": "utf-8"}, offset, limit)
    if isinstance(sliced, ReadResult):
        return sliced
    return ReadResult(file_data={"content": sliced, "encoding": "utf-8"})


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
    """Filter listed blobs to those inside *search_path* matching *glob*."""
    candidates: list[Any] = []
    for blob in blobs:
        virtual = from_blob_key(prefix, blob.name)
        relative = _relative_path(virtual, search_path)
        if relative is None:
            continue
        if glob and not wcglob.globmatch(relative, glob, flags=_GLOB_FLAGS):
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


def _blob_metadata(created_at: str | None = None) -> dict[str, str]:
    """Build blob metadata with created/modified timestamps."""
    now = datetime.now(timezone.utc).isoformat()
    return {"created_at": created_at or now, "modified_at": now}


class AzureBlobBackend(BackendProtocol):
    """Azure Blob Storage filesystem backend for Deep Agents.

    Implements ``BackendProtocol`` using Azure Blob Storage as the persistence
    layer. File content is stored in blob bodies (UTF-8 text, or raw bytes for
    binary uploads), with ``created_at``/``modified_at`` timestamps in blob
    metadata. Directories are synthesized on the fly from blob key prefixes (no
    directory marker blobs).
    """

    _MAX_CONCURRENCY = 8

    def __init__(
        self,
        account_url: str = "",
        container_name: str = "",
        *,
        prefix: str | None = None,
        credential: _SDK_CREDENTIAL_TYPE = None,
        connection_string: str | None = None,
    ) -> None:
        """Create a new backend instance.

        Args:
            account_url: Account URL, e.g.
                ``https://<account>.blob.core.windows.net``. Ignored when
                ``connection_string`` is provided.
            container_name: Target blob container name.
            prefix: Optional key namespace prefix within the container. Scoping
                each agent/session to a prefix isolates their files.
            credential: Credential to authenticate with. If ``None``,
                ``DefaultAzureCredential`` is used. Ignored when
                ``connection_string`` is provided.
            connection_string: Full connection string (e.g. from the Azure
                portal, or for the Azurite emulator). Used instead of
                ``account_url`` + ``credential`` when set.
        """
        self._account_url = account_url
        self._container_name = container_name
        self._prefix = prefix or ""
        self._credential = credential
        self._connection_string = connection_string

    # TODO: These credential helpers are identical (modulo error-message text)
    # to AzureBlobStorageLoader's in document_loaders.py. Consolidate into a
    # shared module in a follow-up PR.
    def _get_sync_credential(
        self, provided_credential: _SDK_CREDENTIAL_TYPE
    ) -> _SDK_CREDENTIAL_TYPE:
        if provided_credential is None:
            return azure.identity.DefaultAzureCredential()
        if isinstance(
            provided_credential, azure.core.credentials_async.AsyncTokenCredential
        ):
            raise ValueError(
                "Cannot use synchronous methods when AzureBlobBackend is "
                "instantiated with an AsyncTokenCredential. Use the async "
                "methods instead, or supply a synchronous credential."
            )
        return provided_credential

    @asynccontextmanager
    async def _get_async_credential(
        self, provided_credential: _SDK_CREDENTIAL_TYPE
    ) -> AsyncIterator[_SDK_CREDENTIAL_TYPE]:
        if provided_credential is None:
            cred = azure.identity.aio.DefaultAzureCredential()
            async with cred:
                yield cred
        elif not isinstance(
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
        else:
            yield provided_credential

    def _client_kwargs(self, credential: Any) -> dict[str, Any]:
        return {
            "account_url": self._account_url,
            "container_name": self._container_name,
            "credential": credential,
            "user_agent": USER_AGENT,
        }

    @contextmanager
    def _sync_container(self) -> Iterator[ContainerClient]:
        if self._connection_string:
            with ContainerClient.from_connection_string(
                self._connection_string,
                self._container_name,
                user_agent=USER_AGENT,
            ) as container:
                yield container
            return

        if self._credential is None:
            # We created this credential, so we close it. A caller-supplied
            # credential is caller-owned and is never closed here.
            credential = azure.identity.DefaultAzureCredential()
            try:
                with ContainerClient(**self._client_kwargs(credential)) as container:
                    yield container
            finally:
                credential.close()
        else:
            sync_credential = self._get_sync_credential(self._credential)
            with ContainerClient(**self._client_kwargs(sync_credential)) as container:
                yield container

    @asynccontextmanager
    async def _async_container(self) -> AsyncIterator[AsyncContainerClient]:
        if self._connection_string:
            async with AsyncContainerClient.from_connection_string(
                self._connection_string,
                self._container_name,
                user_agent=USER_AGENT,
            ) as container:
                yield container
            return

        async with self._get_async_credential(self._credential) as credential:
            async with AsyncContainerClient(
                **self._client_kwargs(credential)
            ) as container:
                yield container

    def ls(self, path: str) -> LsResult:
        """List files and synthesized subdirectories at a path.

        Subdirectories are synthesized from blob key prefixes; no directory
        marker blobs are required.

        Args:
            path: Virtual directory path (e.g. ``"/src"``).

        Returns:
            An ``LsResult`` whose ``entries`` holds the immediate children.
        """
        try:
            normalized_root = validate_path(path or "/")
        except ValueError:
            return LsResult(entries=[])

        with self._sync_container() as container:
            blobs = [
                blob
                for blob in container.list_blobs(
                    name_starts_with=get_prefix_for_path(self._prefix, normalized_root)
                    or None,
                    include=["metadata"],
                )
            ]
        return _ls_result(blobs, self._prefix, normalized_root)

    async def als(self, path: str) -> LsResult:
        """List files and synthesized subdirectories at a path.

        Subdirectories are synthesized from blob key prefixes; no directory
        marker blobs are required.

        Args:
            path: Virtual directory path (e.g. ``"/src"``).

        Returns:
            An ``LsResult`` whose ``entries`` holds the immediate children.
        """
        try:
            normalized_root = validate_path(path or "/")
        except ValueError:
            return LsResult(entries=[])

        async with self._async_container() as container:
            blobs = [
                blob
                async for blob in container.list_blobs(
                    name_starts_with=get_prefix_for_path(self._prefix, normalized_root)
                    or None,
                    include=["metadata"],
                )
            ]
        return _ls_result(blobs, self._prefix, normalized_root)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a file and return its raw content for the requested window.

        The content is returned unformatted: the Deep Agents middleware applies
        line numbering, the empty-file reminder, and base64 multimodal handling
        based on the ``encoding`` field. Blobs that are not valid UTF-8 are
        returned base64-encoded with ``encoding="base64"``.

        Args:
            file_path: Virtual path to the file.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return.

        Returns:
            A ``ReadResult`` with the file content, or an error if the file is
            not found or the offset is out of range.
        """
        try:
            file_path = validate_path(file_path)
        except ValueError as exc:
            return ReadResult(error=f"Error: Invalid path '{file_path}': {exc}")

        with self._sync_container() as container:
            try:
                raw = (
                    container.get_blob_client(self._blob_key(file_path))
                    .download_blob()
                    .readall()
                )
            except ResourceNotFoundError:
                return ReadResult(error=f"Error: File '{file_path}' not found")
        return _read_result_from_bytes(raw, offset, limit)

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        """Read a file and return its raw content for the requested window.

        The content is returned unformatted: the Deep Agents middleware applies
        line numbering, the empty-file reminder, and base64 multimodal handling
        based on the ``encoding`` field. Blobs that are not valid UTF-8 are
        returned base64-encoded with ``encoding="base64"``.

        Args:
            file_path: Virtual path to the file.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return.

        Returns:
            A ``ReadResult`` with the file content, or an error if the file is
            not found or the offset is out of range.
        """
        try:
            file_path = validate_path(file_path)
        except ValueError as exc:
            return ReadResult(error=f"Error: Invalid path '{file_path}': {exc}")

        async with self._async_container() as container:
            try:
                stream = await container.get_blob_client(
                    self._blob_key(file_path)
                ).download_blob()
                raw = await stream.readall()
            except ResourceNotFoundError:
                return ReadResult(error=f"Error: File '{file_path}' not found")
        return _read_result_from_bytes(raw, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Create a new file with the given content.

        Fails if the file already exists; use ``edit``/``aedit`` to modify an
        existing file.

        Args:
            file_path: Virtual path for the new file.
            content: UTF-8 text content to write.

        Returns:
            A ``WriteResult`` with the path, or an error if the file exists.
        """
        try:
            file_path = validate_path(file_path)
        except ValueError as exc:
            return WriteResult(error=f"Invalid path '{file_path}': {exc}")

        with self._sync_container() as container:
            try:
                container.get_blob_client(self._blob_key(file_path)).upload_blob(
                    content.encode("utf-8"),
                    overwrite=False,
                    metadata=_blob_metadata(),
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
            A ``WriteResult`` with the path, or an error if the file exists.
        """
        try:
            file_path = validate_path(file_path)
        except ValueError as exc:
            return WriteResult(error=f"Invalid path '{file_path}': {exc}")

        async with self._async_container() as container:
            try:
                await container.get_blob_client(self._blob_key(file_path)).upload_blob(
                    content.encode("utf-8"),
                    overwrite=False,
                    metadata=_blob_metadata(),
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
            if the file is missing or the match is not unique.
        """
        try:
            file_path = validate_path(file_path)
        except ValueError as exc:
            return EditResult(error=f"Invalid path '{file_path}': {exc}")

        blob_key = self._blob_key(file_path)
        with self._sync_container() as container:
            blob = container.get_blob_client(blob_key)
            try:
                content = str(blob.download_blob(encoding="utf-8").readall())
                metadata = dict(blob.get_blob_properties().metadata or {})
            except ResourceNotFoundError:
                return EditResult(error=f"Error: File '{file_path}' not found")

            result = perform_string_replacement(
                content, old_string, new_string, replace_all
            )
            if isinstance(result, str):
                return EditResult(error=result)
            new_content, occurrences = result
            blob.upload_blob(
                new_content.encode("utf-8"),
                overwrite=True,
                metadata=_blob_metadata(metadata.get("created_at")),
            )
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
            if the file is missing or the match is not unique.
        """
        try:
            file_path = validate_path(file_path)
        except ValueError as exc:
            return EditResult(error=f"Invalid path '{file_path}': {exc}")

        blob_key = self._blob_key(file_path)
        async with self._async_container() as container:
            blob = container.get_blob_client(blob_key)
            try:
                stream = await blob.download_blob(encoding="utf-8")
                content = str(await stream.readall())
                props = await blob.get_blob_properties()
                metadata = dict(props.metadata or {})
            except ResourceNotFoundError:
                return EditResult(error=f"Error: File '{file_path}' not found")

            result = perform_string_replacement(
                content, old_string, new_string, replace_all
            )
            if isinstance(result, str):
                return EditResult(error=result)
            new_content, occurrences = result
            await blob.upload_blob(
                new_content.encode("utf-8"),
                overwrite=True,
                metadata=_blob_metadata(metadata.get("created_at")),
            )
        return EditResult(path=file_path, occurrences=int(occurrences))

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern.

        Supports ``**`` (globstar) and ``{a,b}`` brace expansion. The pattern is
        matched against paths relative to *path*.

        Args:
            pattern: Glob pattern (e.g. ``"**/*.py"``).
            path: Base directory for the search (default: ``"/"``).

        Returns:
            A ``GlobResult`` whose ``matches`` holds the matching files.
        """
        try:
            normalized_path = validate_path(path or "/")
        except ValueError:
            return GlobResult(matches=[])

        with self._sync_container() as container:
            blobs = self._list_target_blobs_sync(container, normalized_path)
        return _glob_result(blobs, self._prefix, normalized_path, pattern)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern.

        Supports ``**`` (globstar) and ``{a,b}`` brace expansion. The pattern is
        matched against paths relative to *path*.

        Args:
            pattern: Glob pattern (e.g. ``"**/*.py"``).
            path: Base directory for the search (default: ``"/"``).

        Returns:
            A ``GlobResult`` whose ``matches`` holds the matching files.
        """
        try:
            normalized_path = validate_path(path or "/")
        except ValueError:
            return GlobResult(matches=[])

        async with self._async_container() as container:
            blobs = await self._list_target_blobs_async(container, normalized_path)
        return _glob_result(blobs, self._prefix, normalized_path, pattern)

    def grep(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> GrepResult:
        """Search file contents for a literal substring.

        Args:
            pattern: Literal substring to search for.
            path: Directory scope for the search (default: ``"/"``).
            glob: Optional relative-path glob to pre-filter files (e.g.
                ``"*.py"``).

        Returns:
            A ``GrepResult`` whose ``matches`` holds matching lines, or whose
            ``error`` is set when the path is invalid or a file cannot be read.
        """
        try:
            search_path = validate_path(path or "/")
        except ValueError as exc:
            invalid = path if path is not None else "/"
            return GrepResult(error=f"Error: Invalid path '{invalid}': {exc}")

        matches: list[GrepMatch] = []
        failed: list[str] = []
        with self._sync_container() as container:
            blobs = self._list_target_blobs_sync(container, search_path)
            for blob in _grep_candidates(blobs, self._prefix, search_path, glob):
                virtual = from_blob_key(self._prefix, blob.name)
                try:
                    content = str(
                        container.get_blob_client(blob.name)
                        .download_blob(encoding="utf-8")
                        .readall()
                    )
                except (AzureError, UnicodeError) as exc:
                    logger.warning(
                        "Failed to read blob %s for grep: %s", blob.name, exc
                    )
                    failed.append(virtual)
                    continue
                matches.extend(_grep_lines(content, pattern, virtual))

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
            glob: Optional relative-path glob to pre-filter files (e.g.
                ``"*.py"``).

        Returns:
            A ``GrepResult`` whose ``matches`` holds matching lines, or whose
            ``error`` is set when the path is invalid or a file cannot be read.
        """
        import asyncio

        try:
            search_path = validate_path(path or "/")
        except ValueError as exc:
            invalid = path if path is not None else "/"
            return GrepResult(error=f"Error: Invalid path '{invalid}': {exc}")

        matches: list[GrepMatch] = []
        failed: list[str] = []
        async with self._async_container() as container:
            blobs = await self._list_target_blobs_async(container, search_path)
            candidates = _grep_candidates(blobs, self._prefix, search_path, glob)
            semaphore = asyncio.Semaphore(self._MAX_CONCURRENCY)

            async def scan(blob: Any) -> list[GrepMatch]:
                virtual = from_blob_key(self._prefix, blob.name)
                async with semaphore:
                    try:
                        stream = await container.get_blob_client(
                            blob.name
                        ).download_blob(encoding="utf-8")
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
        responses: list[FileUploadResponse] = []
        with self._sync_container() as container:
            for file_path, content in files:
                try:
                    validated = validate_path(file_path)
                except ValueError:
                    responses.append(
                        FileUploadResponse(path=file_path, error="invalid_path")
                    )
                    continue
                try:
                    container.get_blob_client(self._blob_key(validated)).upload_blob(
                        content, overwrite=True, metadata=_blob_metadata()
                    )
                    responses.append(FileUploadResponse(path=validated, error=None))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to upload %s: %s", validated, exc)
                    responses.append(
                        FileUploadResponse(path=validated, error="permission_denied")
                    )
        return responses

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload one or more binary files, overwriting any that exist.

        Args:
            files: List of ``(path, content_bytes)`` tuples.

        Returns:
            A list of ``FileUploadResponse`` objects, one per input file.
        """
        responses: list[FileUploadResponse] = []
        async with self._async_container() as container:
            for file_path, content in files:
                try:
                    validated = validate_path(file_path)
                except ValueError:
                    responses.append(
                        FileUploadResponse(path=file_path, error="invalid_path")
                    )
                    continue
                try:
                    await container.get_blob_client(
                        self._blob_key(validated)
                    ).upload_blob(content, overwrite=True, metadata=_blob_metadata())
                    responses.append(FileUploadResponse(path=validated, error=None))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to upload %s: %s", validated, exc)
                    responses.append(
                        FileUploadResponse(path=validated, error="permission_denied")
                    )
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download one or more files as raw bytes.

        Args:
            paths: Virtual paths to download.

        Returns:
            A list of ``FileDownloadResponse`` objects, one per input path;
            each has ``content`` on success or ``error="file_not_found"``.
        """
        responses: list[FileDownloadResponse] = []
        with self._sync_container() as container:
            for file_path in paths:
                try:
                    validated = validate_path(file_path)
                except ValueError:
                    responses.append(
                        FileDownloadResponse(
                            path=file_path, content=None, error="invalid_path"
                        )
                    )
                    continue
                try:
                    raw = (
                        container.get_blob_client(self._blob_key(validated))
                        .download_blob()
                        .readall()
                    )
                    responses.append(
                        FileDownloadResponse(
                            path=validated, content=_as_bytes(raw), error=None
                        )
                    )
                except ResourceNotFoundError:
                    responses.append(
                        FileDownloadResponse(
                            path=validated, content=None, error="file_not_found"
                        )
                    )
        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download one or more files as raw bytes.

        Args:
            paths: Virtual paths to download.

        Returns:
            A list of ``FileDownloadResponse`` objects, one per input path;
            each has ``content`` on success or ``error="file_not_found"``.
        """
        responses: list[FileDownloadResponse] = []
        async with self._async_container() as container:
            for file_path in paths:
                try:
                    validated = validate_path(file_path)
                except ValueError:
                    responses.append(
                        FileDownloadResponse(
                            path=file_path, content=None, error="invalid_path"
                        )
                    )
                    continue
                try:
                    stream = await container.get_blob_client(
                        self._blob_key(validated)
                    ).download_blob()
                    raw = await stream.readall()
                    responses.append(
                        FileDownloadResponse(
                            path=validated, content=_as_bytes(raw), error=None
                        )
                    )
                except ResourceNotFoundError:
                    responses.append(
                        FileDownloadResponse(
                            path=validated, content=None, error="file_not_found"
                        )
                    )
        return responses

    def _blob_key(self, path: str) -> str:
        return to_blob_key(self._prefix, path)

    def _list_target_blobs_sync(
        self, container: ContainerClient, path: str
    ) -> list[Any]:
        if path == "/":
            return self._list_blobs_sync(container, to_blob_key(self._prefix, "/"))

        blob = container.get_blob_client(self._blob_key(path))
        try:
            props = blob.get_blob_properties()
        except ResourceNotFoundError:
            props = None
        if props is not None:
            metadata = dict(props.metadata) if props.metadata else None
            return [
                _blob_view(self._blob_key(path), getattr(props, "size", 0), metadata)
            ]

        return self._list_blobs_sync(container, get_prefix_for_path(self._prefix, path))

    async def _list_target_blobs_async(
        self, container: AsyncContainerClient, path: str
    ) -> list[Any]:
        if path == "/":
            return await self._list_blobs_async(
                container, to_blob_key(self._prefix, "/")
            )

        blob = container.get_blob_client(self._blob_key(path))
        try:
            props = await blob.get_blob_properties()
        except ResourceNotFoundError:
            props = None
        if props is not None:
            metadata = dict(props.metadata) if props.metadata else None
            return [
                _blob_view(self._blob_key(path), getattr(props, "size", 0), metadata)
            ]

        return await self._list_blobs_async(
            container, get_prefix_for_path(self._prefix, path)
        )

    def _list_blobs_sync(self, container: ContainerClient, prefix: str) -> list[Any]:
        return [
            blob
            for blob in container.list_blobs(
                name_starts_with=prefix or None, include=["metadata"]
            )
        ]

    async def _list_blobs_async(
        self, container: AsyncContainerClient, prefix: str
    ) -> list[Any]:
        return [
            blob
            async for blob in container.list_blobs(
                name_starts_with=prefix or None, include=["metadata"]
            )
        ]


def _already_exists_error(file_path: str) -> str:
    return (
        f"Cannot write to {file_path} because it already exists. "
        f"Read and then make an edit, or write to a new path."
    )


def _as_bytes(raw: Any) -> bytes:
    return raw if isinstance(raw, bytes) else raw.encode("utf-8")
