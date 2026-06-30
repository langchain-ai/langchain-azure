"""Azure Blob Storage backend for LangChain Deep Agents."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional

import wcmatch.glob as wcglob
from azure.core.credentials import AzureSasCredential
from azure.core.exceptions import (
    AzureError,
    ResourceExistsError,
    ResourceNotFoundError,
)
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
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

from ._path import from_blob_key, get_prefix_for_path, to_blob_key
from ._utils import build_file_info
from .config import AzureBlobConfig

logger = logging.getLogger(__name__)


@dataclass
class _ClientState:
    client: Optional[BlobServiceClient] = None
    container: Optional[ContainerClient] = None
    credential: Optional[Any] = None
    init_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    user_credential: Optional[Any] = None


_temporary_client_states: contextvars.ContextVar[dict[int, _ClientState] | None] = (
    contextvars.ContextVar(
        "langchain_azure_storage_deepagents_temporary_client_states",
        default=None,
    )
)


class AzureBlobBackend(BackendProtocol):
    """Azure Blob Storage filesystem backend for Deep Agents.

    Implements BackendProtocol using Azure Blob Storage as the persistence
    layer. All file content is stored as raw UTF-8 text in blob bodies, with
    ``created_at`` and ``modified_at`` timestamps in blob metadata.

    Directories are synthesized on-the-fly from blob key prefixes (no
    directory marker blobs).

    Native async implementation; sync methods delegate to async via
    ``asyncio.run()``.
    """

    def __init__(self, config: AzureBlobConfig) -> None:
        """Create a new backend instance.

        The Azure SDK clients are **not** created here; they are lazily
        initialized on the first operation.  Call ``close`` (or
        ``await close()``) when done to release network resources.

        Args:
            config: Backend configuration (connection details, prefix, etc.).
        """
        self._config = config
        self._client: Optional[BlobServiceClient] = None
        self._container: Optional[ContainerClient] = None
        self._credential: Optional[Any] = None
        self._init_lock = asyncio.Lock()

    async def _get_container(self) -> ContainerClient:
        """Lazily initialise and return the container client."""
        temporary_states = _temporary_client_states.get()
        if temporary_states is None:
            state = None
            if self._container is not None:
                return self._container
            init_lock = self._init_lock
        else:
            state = temporary_states.get(id(self))
            if state is None:
                state = _ClientState(user_credential=self._config.credential)
                temporary_states[id(self)] = state
            if state.container is not None:
                return state.container
            init_lock = state.init_lock

        async with init_lock:
            # Double-checked locking to avoid races when lazily initialising.
            if state is None:
                if self._container is not None:
                    return self._container
            elif state.container is not None:
                return state.container

            kwargs: dict[str, Any] = {"user_agent": USER_AGENT}
            if self._config.api_version:
                kwargs["api_version"] = self._config.api_version

            stored_credential: Any | None = None
            credential: Any
            if self._config.connection_string:
                client = BlobServiceClient.from_connection_string(
                    self._config.connection_string,
                    **kwargs,
                )
            elif self._config.account_key:
                client = BlobServiceClient(
                    account_url=self._config.account_url,
                    credential=self._config.account_key,
                    **kwargs,
                )
            elif self._config.sas_token:
                credential = AzureSasCredential(self._config.sas_token)
                client = BlobServiceClient(
                    account_url=self._config.account_url,
                    credential=credential,
                    **kwargs,
                )
            elif self._config.credential is not None:
                # A caller-supplied credential is caller-owned: never stored for
                # closing here (close() would otherwise tear down a credential
                # the caller may share with other Azure clients).
                credential = self._config.credential
                client = BlobServiceClient(
                    account_url=self._config.account_url,
                    credential=credential,
                    **kwargs,
                )
            else:
                from azure.identity.aio import DefaultAzureCredential

                credential = DefaultAzureCredential()
                stored_credential = credential
                client = BlobServiceClient(
                    account_url=self._config.account_url,
                    credential=credential,
                    **kwargs,
                )

            container = client.get_container_client(self._config.container_name)
            if state is None:
                self._client = client
                self._credential = stored_credential
                self._container = container
            else:
                state.client = client
                state.credential = stored_credential
                state.container = container
            return container

    async def close(self) -> None:
        """Close the underlying Azure SDK clients and release resources.

        Safe to call multiple times. After closing, subsequent operations will
        lazily re-initialize a new client.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._container = None
        if self._credential is not None:
            await self._credential.close()
            self._credential = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _blob_key(self, path: str) -> str:
        return to_blob_key(self._config.prefix, path)

    def _virtual_path(self, blob_name: str) -> str:
        return from_blob_key(self._config.prefix, blob_name)

    def _validate_file_path(self, path: str) -> str:
        return validate_path(path)

    def _validate_search_path(self, path: str | None) -> str:
        return validate_path(path or "/")

    def _relative_path(self, virtual_path: str, base_path: str) -> str | None:
        if base_path == "/":
            return virtual_path[1:]

        prefix_with_slash = base_path + "/"
        if virtual_path.startswith(prefix_with_slash):
            return virtual_path[len(prefix_with_slash) :]
        if virtual_path == base_path:
            return virtual_path.split("/")[-1]
        return None

    async def _get_listed_blob(
        self, container: ContainerClient, blob_key: str
    ) -> Any | None:
        """Return a blob-like object for an exact key if it exists."""
        blob = container.get_blob_client(blob_key)
        try:
            props = await blob.get_blob_properties()
        except ResourceNotFoundError:
            return None

        metadata = dict(props.metadata) if props.metadata else None
        return SimpleNamespace(
            name=blob_key,
            size=getattr(props, "size", 0),
            metadata=metadata,
        )

    async def _list_target_blobs(
        self, container: ContainerClient, path: str
    ) -> list[Any]:
        """List blobs for a search path, preferring an exact file match."""
        if path == "/":
            return await self._list_blobs(
                container, to_blob_key(self._config.prefix, "/")
            )

        exact_blob = await self._get_listed_blob(container, self._blob_key(path))
        if exact_blob is not None:
            return [exact_blob]

        return await self._list_blobs(
            container, get_prefix_for_path(self._config.prefix, path)
        )

    async def _blob_exists(self, container: ContainerClient, blob_key: str) -> bool:
        blob = container.get_blob_client(blob_key)
        return await blob.exists()

    async def _read_blob(
        self, container: ContainerClient, blob_key: str
    ) -> tuple[str, dict[str, str]]:
        """Download blob content and metadata.

        Returns:
            Tuple of (content_string, metadata_dict).

        Raises:
            ResourceNotFoundError: If blob does not exist.
        """
        blob = container.get_blob_client(blob_key)
        stream = await blob.download_blob(encoding=self._config.encoding)
        content = str(await stream.readall())
        props = await blob.get_blob_properties()
        metadata: dict[str, str] = dict(props.metadata) if props.metadata else {}
        return content, metadata

    async def _write_blob(
        self,
        container: ContainerClient,
        blob_key: str,
        content: str,
        *,
        created_at: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        """Upload content to a blob with timestamps in metadata."""
        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "created_at": created_at or now,
            "modified_at": now,
        }
        blob = container.get_blob_client(blob_key)
        await blob.upload_blob(
            content.encode(self._config.encoding),
            overwrite=overwrite,
            metadata=metadata,
        )

    async def _list_blobs(
        self,
        container: ContainerClient,
        prefix: str,
    ) -> list[Any]:
        """List all blobs under the given prefix."""
        blobs = []
        async for blob in container.list_blobs(
            name_starts_with=prefix or None,
            include=["metadata"],
        ):
            blobs.append(blob)
        return blobs

    # ------------------------------------------------------------------
    # Sync wrappers
    # ------------------------------------------------------------------

    def _run_async(self, coro_factory: Callable[[], Awaitable[Any]]) -> Any:
        """Run an async coroutine from sync context.

        Sync wrappers run the coroutine in a fresh event loop. Azure async
        clients are loop-affine, so ``_get_container`` uses context-local
        temporary client state while the coroutine runs instead of reusing or
        mutating instance-level async caches. The coroutine is constructed
        after the context is installed so Python versions that bind coroutine
        context at creation still see the temporary state.

        A user-supplied credential (``config.credential``) is treated as
        caller-owned and is never closed here. If that credential is an async
        Azure Identity credential that has already been used in another loop,
        the sync wrappers cannot make it cross-loop safe. For sync-wrapper use,
        prefer ``connection_string``, ``account_key``, ``sas_token``, or omit
        ``credential`` so the backend creates a fresh ``DefaultAzureCredential``
        inside each temporary loop.
        """

        async def _run_and_cleanup() -> Any:
            states: dict[int, _ClientState] = {}
            token = _temporary_client_states.set(states)
            try:
                return await coro_factory()
            finally:
                for state in states.values():
                    if state.client is not None:
                        try:
                            await state.client.close()
                        except Exception:  # noqa: BLE001
                            logger.warning(
                                "Error closing temporary client", exc_info=True
                            )
                    # Only close credentials we created ourselves
                    # (e.g. DefaultAzureCredential). Skip user-supplied
                    # credentials so we don't tear down caller-owned state.
                    if (
                        state.credential is not None
                        and state.credential is not state.user_credential
                    ):
                        try:
                            await state.credential.close()
                        except Exception:  # noqa: BLE001
                            logger.warning(
                                "Error closing temporary credential",
                                exc_info=True,
                            )
                _temporary_client_states.reset(token)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an event loop -- cannot use asyncio.run().
            # Create a new thread with its own loop.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _run_and_cleanup())
                return future.result()
        return asyncio.run(_run_and_cleanup())

    # ------------------------------------------------------------------
    # ls
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        """List files and subdirectories at *path*."""
        return self._run_async(lambda: self.als(path))

    async def als(self, path: str) -> LsResult:
        """List files and subdirectories at *path*.

        Directories are synthesized from blob key prefixes -- no marker blobs
        are required.

        Args:
            path: Virtual directory path (e.g., ``"/src"``).

        Returns:
            Sorted list of ``FileInfo`` dicts for immediate children.
        """
        try:
            normalized_root = self._validate_search_path(path)
        except ValueError:
            return LsResult(entries=[])

        container = await self._get_container()
        blobs = await self._list_blobs(
            container,
            get_prefix_for_path(self._config.prefix, normalized_root),
        )
        if not blobs:
            return LsResult(entries=[])

        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # Normalize the virtual directory path.
        normalized_path = (
            normalized_root if normalized_root.endswith("/") else normalized_root + "/"
        )

        for blob in blobs:
            virtual = self._virtual_path(blob.name)

            # Get relative path from the listing directory.
            if not virtual.startswith(normalized_path):
                continue

            relative = virtual[len(normalized_path) :]
            if not relative:
                continue

            if "/" in relative:
                # Subdirectory -- extract immediate child dir name.
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
            else:
                # Direct file in this directory.
                modified_at = ""
                if blob.metadata:
                    modified_at = blob.metadata.get("modified_at", "")
                infos.append(
                    build_file_info(
                        path=virtual,
                        is_dir=False,
                        size=blob.size or 0,
                        modified_at=modified_at,
                    )
                )

        # Add synthesized directory entries.
        for subdir in sorted(subdirs):
            infos.append(build_file_info(path=subdir, is_dir=True, size=0))

        infos.sort(key=lambda x: x.get("path", ""))
        return LsResult(entries=infos)

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a file and return its content with line numbers.

        Sync wrapper for ``aread``.
        """
        return self._run_async(lambda: self.aread(file_path, offset, limit))

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        """Read a file and return its content with line numbers.

        Args:
            file_path: Virtual path to the file.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return.

        Returns:
            ``ReadResult`` with raw (unformatted) file content on success, or an
            error if the file is not found or the offset is out of range.

            The returned content is **not** line-number formatted: the Deep
            Agents filesystem middleware applies line numbering, the empty-file
            reminder, and base64 multimodal handling at the tool boundary based
            on the ``encoding`` field. Binary blobs (content that is not valid
            UTF-8) are returned base64-encoded with ``encoding="base64"``.
        """
        try:
            file_path = self._validate_file_path(file_path)
        except ValueError as exc:
            return ReadResult(error=f"Error: Invalid path '{file_path}': {exc}")

        container = await self._get_container()
        blob_key = self._blob_key(file_path)

        try:
            blob = container.get_blob_client(blob_key)
            stream = await blob.download_blob()
            raw = await stream.readall()
        except ResourceNotFoundError:
            return ReadResult(error=f"Error: File '{file_path}' not found")

        content_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
        try:
            text = content_bytes.decode(self._config.encoding)
        except UnicodeDecodeError:
            # Binary content: hand it to the middleware base64-encoded so it can
            # be returned as a multimodal content block. Binary reads are not
            # line-sliced.
            b64 = base64.b64encode(content_bytes).decode("ascii")
            return ReadResult(file_data={"content": b64, "encoding": "base64"})

        sliced = slice_read_response(
            {"content": text, "encoding": "utf-8"}, offset, limit
        )
        if isinstance(sliced, ReadResult):
            return sliced
        return ReadResult(file_data={"content": sliced, "encoding": "utf-8"})

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        """Create a new file (sync wrapper for ``awrite``)."""
        return self._run_async(lambda: self.awrite(file_path, content))

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Create a new file with the given content.

        Fails if the file already exists -- use ``aedit`` to modify existing
        files.

        Args:
            file_path: Virtual path for the new file.
            content: UTF-8 text content to write.

        Returns:
            ``WriteResult`` with the path on success, or an error if the file
            already exists.
        """
        try:
            file_path = self._validate_file_path(file_path)
        except ValueError as exc:
            return WriteResult(error=f"Invalid path '{file_path}': {exc}")

        container = await self._get_container()
        blob_key = self._blob_key(file_path)

        try:
            await self._write_blob(container, blob_key, content, overwrite=False)
        except ResourceExistsError:
            return WriteResult(
                error=(
                    f"Cannot write to {file_path} because it already exists. "
                    f"Read and then make an edit, or write to a new path."
                )
            )
        return WriteResult(path=file_path)

    # ------------------------------------------------------------------
    # edit
    # ------------------------------------------------------------------

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace text in an existing file (sync wrapper for ``aedit``)."""
        return self._run_async(
            lambda: self.aedit(file_path, old_string, new_string, replace_all)
        )

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
            old_string: The exact substring to find.
            new_string: The replacement text.
            replace_all: If ``True``, replace every occurrence; otherwise
                require exactly one match.

        Returns:
            ``EditResult`` with the path and occurrence count on success, or an
            error string if the file is not found or the match is ambiguous.
        """
        try:
            file_path = self._validate_file_path(file_path)
        except ValueError as exc:
            return EditResult(error=f"Invalid path '{file_path}': {exc}")

        container = await self._get_container()
        blob_key = self._blob_key(file_path)

        try:
            content, metadata = await self._read_blob(container, blob_key)
        except ResourceNotFoundError:
            return EditResult(error=f"Error: File '{file_path}' not found")

        result = perform_string_replacement(
            content, old_string, new_string, replace_all
        )
        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        created_at = metadata.get("created_at")
        await self._write_blob(container, blob_key, new_content, created_at=created_at)
        return EditResult(path=file_path, occurrences=int(occurrences))

    # ------------------------------------------------------------------
    # glob
    # ------------------------------------------------------------------

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern."""
        return self._run_async(lambda: self.aglob(pattern, path))

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern.

        Supports ``**`` (globstar) and ``{a,b}`` brace expansion via
        *wcmatch*.

        Args:
            pattern: Glob pattern relative to *path* (e.g., ``"**/*.py"``).
            path: Base directory for the search.

        Returns:
            A ``GlobResult`` whose ``matches`` field contains the matching
            ``FileInfo`` dicts.
        """
        try:
            normalized_path = self._validate_search_path(path or "/")
        except ValueError:
            return GlobResult(matches=[])

        container = await self._get_container()
        blobs = await self._list_target_blobs(container, normalized_path)
        if not blobs:
            return GlobResult(matches=[])

        infos: list[FileInfo] = []
        for blob in blobs:
            virtual = self._virtual_path(blob.name)

            relative = self._relative_path(virtual, normalized_path)
            if relative is None:
                continue

            if wcglob.globmatch(
                relative, pattern, flags=wcglob.BRACE | wcglob.GLOBSTAR
            ):
                modified_at = ""
                if blob.metadata:
                    modified_at = blob.metadata.get("modified_at", "")
                infos.append(
                    build_file_info(
                        path=virtual,
                        is_dir=False,
                        size=blob.size or 0,
                        modified_at=modified_at,
                    )
                )

        return GlobResult(matches=infos)

    # ------------------------------------------------------------------
    # grep
    # ------------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search file contents for a literal substring."""
        return self._run_async(lambda: self.agrep(pattern, path, glob))

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search file contents for a literal substring.

        Blobs are downloaded and scanned concurrently (bounded by
        ``AzureBlobConfig.max_concurrency``).

        Args:
            pattern: Literal substring to search for.
            path: Directory scope for the search (default: ``"/"``).
            glob: Optional relative-path glob to pre-filter blobs (e.g.,
                ``"*.py"`` for direct children or ``"src/**/*.py"``
                from the search root).

        Returns:
            A ``GrepResult`` whose ``matches`` field contains any matching
            ``GrepMatch`` dicts and whose ``error`` field is set when the
            search path is invalid or a blob cannot be read reliably.
        """
        try:
            search_path = self._validate_search_path(path)
        except ValueError as exc:
            invalid_path = path if path is not None else "/"
            return GrepResult(error=f"Error: Invalid path '{invalid_path}': {exc}")

        container = await self._get_container()
        blobs = await self._list_target_blobs(container, search_path)
        if not blobs:
            return GrepResult(matches=[])

        blob_candidates: list[tuple[Any, str]] = []
        for blob in blobs:
            virtual = self._virtual_path(blob.name)
            relative = self._relative_path(virtual, search_path)
            if relative is None:
                continue
            blob_candidates.append((blob, relative))

        # Filter by glob pattern on relative path if provided.
        if glob:
            blob_candidates = [
                (blob, relative)
                for blob, relative in blob_candidates
                if wcglob.globmatch(
                    relative,
                    glob,
                    flags=wcglob.BRACE | wcglob.GLOBSTAR,
                )
            ]

        matches: list[GrepMatch] = []
        failed_blobs: list[str] = []

        # Process blobs concurrently with bounded concurrency.
        semaphore = asyncio.Semaphore(self._config.max_concurrency)

        async def search_blob(blob: Any) -> list[GrepMatch]:
            async with semaphore:
                try:
                    blob_client = container.get_blob_client(blob.name)
                    stream = await blob_client.download_blob(
                        encoding=self._config.encoding,
                    )
                    content = str(await stream.readall())
                except (AzureError, UnicodeError) as exc:
                    logger.warning(
                        "Failed to read blob %s for grep: %s", blob.name, exc
                    )
                    failed_blobs.append(self._virtual_path(blob.name))
                    return []

                virtual = self._virtual_path(blob.name)
                blob_matches: list[GrepMatch] = []
                for line_num, line in enumerate(content.split("\n"), 1):
                    if pattern in line:  # Literal substring match.
                        blob_matches.append(
                            {"path": virtual, "line": line_num, "text": line}
                        )
                return blob_matches

        results = await asyncio.gather(
            *(search_blob(blob) for blob, _ in blob_candidates)
        )
        for blob_matches in results:
            matches.extend(blob_matches)

        if failed_blobs:
            failed_blobs.sort()
            sample = ", ".join(failed_blobs[:3])
            remainder = len(failed_blobs) - min(len(failed_blobs), 3)
            suffix = f", and {remainder} more" if remainder else ""
            return GrepResult(
                error=(
                    f"Error: grep could not read {len(failed_blobs)} "
                    f"file(s): {sample}{suffix}"
                )
            )

        return GrepResult(matches=matches)

    # ------------------------------------------------------------------
    # upload_files / download_files
    # ------------------------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload binary files (sync wrapper for ``aupload_files``)."""
        return self._run_async(lambda: self.aupload_files(files))

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload one or more binary files, overwriting if they exist.

        Args:
            files: List of ``(path, content_bytes)`` tuples.

        Returns:
            List of ``FileUploadResponse`` dicts, one per input file.
        """
        container = await self._get_container()
        responses: list[FileUploadResponse] = []

        for file_path, content in files:
            try:
                file_path = self._validate_file_path(file_path)
            except ValueError:
                responses.append(
                    FileUploadResponse(path=file_path, error="invalid_path")
                )
                continue

            blob_key = self._blob_key(file_path)
            now = datetime.now(timezone.utc).isoformat()
            metadata = {"created_at": now, "modified_at": now}

            try:
                blob = container.get_blob_client(blob_key)
                await blob.upload_blob(content, overwrite=True, metadata=metadata)
                responses.append(FileUploadResponse(path=file_path, error=None))
            except Exception as exc:
                logger.error("Failed to upload %s: %s", file_path, exc)
                responses.append(
                    FileUploadResponse(path=file_path, error="permission_denied")
                )

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files as raw bytes (sync wrapper for ``adownload_files``)."""
        return self._run_async(lambda: self.adownload_files(paths))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download one or more files as raw bytes.

        Args:
            paths: Virtual paths to download.

        Returns:
            List of ``FileDownloadResponse`` dicts. Each contains ``content``
            (bytes) on success, or ``error`` set to ``"file_not_found"``.
        """
        container = await self._get_container()
        responses: list[FileDownloadResponse] = []

        for file_path in paths:
            try:
                file_path = self._validate_file_path(file_path)
            except ValueError:
                responses.append(
                    FileDownloadResponse(
                        path=file_path, content=None, error="invalid_path"
                    )
                )
                continue

            blob_key = self._blob_key(file_path)
            try:
                blob = container.get_blob_client(blob_key)
                stream = await blob.download_blob()
                raw = await stream.readall()
                content_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
                responses.append(
                    FileDownloadResponse(
                        path=file_path, content=content_bytes, error=None
                    )
                )
            except ResourceNotFoundError:
                responses.append(
                    FileDownloadResponse(
                        path=file_path, content=None, error="file_not_found"
                    )
                )

        return responses
