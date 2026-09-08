"""Azure Container Apps sandboxes backend for deepagents."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING

from azure.core.exceptions import (
    AzureError,
    ResourceNotFoundError,
)
from deepagents.backends.protocol import (
    INVALID_PATH,
    IS_DIRECTORY,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import MAX_BINARY_BYTES, BaseSandbox
from langchain_core._api import beta

from langchain_azure_compute.sandboxes._results import (
    binary_too_large_error,
    execute_error_response,
    execute_response,
    is_text_file,
    read_error,
    read_result_from_bytes,
    transfer_error,
    with_timeout,
    write_error,
)

if TYPE_CHECKING:
    from azure.containerapps.sandbox import FileInfo, SandboxClient
    from azure.containerapps.sandbox.aio import SandboxClient as AsyncSandboxClient

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30 * 60

# Above this size, `read`/`aread` fall back to the base class's server-side
# pagination instead of downloading the whole file for one page: the SDK read
# returns the entire body, and a paginated read of a multi-gigabyte log would
# otherwise transfer all of it -- once per page.
_READ_DOWNLOAD_CAP = 10 * 1024 * 1024

# Identity sentinel `_route_read` returns to mean "use the base class's
# server-side pagination"; never handed to a caller.
_PAGINATE = ReadResult(error="internal routing sentinel")

_BETA_MESSAGE = (
    "`ACASandbox` is in public preview. "
    "Its API is not stable and may change in future versions."
)


def _exists_error(file_path: str, existing: FileInfo) -> str:
    """The `write` refusal message, distinguishing a directory at the path."""
    if existing.is_directory:
        return f"Path '{file_path}' is a directory"
    return (
        f"File '{file_path}' already exists. Use edit to modify it, or delete it first."
    )


@beta(message=_BETA_MESSAGE)
class ACASandbox(BaseSandbox):
    """Deep Agents sandbox backend backed by an Azure Container Apps sandbox.

    Wraps an already-constructed `SandboxClient`, so this class never builds
    endpoints or handles credentials itself. Build the client with the region
    endpoint, an azure-identity credential, and the sandbox's resource
    coordinates:

    ```python
    from azure.containerapps.sandbox import SandboxClient, endpoint_for_region
    from azure.identity import DefaultAzureCredential

    from langchain_azure_compute.sandboxes import ACASandbox

    client = SandboxClient(
        endpoint_for_region("westus2"),
        DefaultAzureCredential(),
        subscription_id="...",
        resource_group="my-rg",
        sandbox_group="my-sandbox-group",
        sandbox_id="...",
    )
    sandbox = ACASandbox(client)
    ```

    The caller keeps ownership of the sandbox lifecycle (`ensure_running`,
    `begin_stop`, snapshots, and so on) via that same client. The one
    exception is `aclose()`, which closes a supplied `async_client`'s
    connection pool -- see its docstring.

    Two operational notes:

    - **Timeouts above the transport's read timeout are unreachable.** The
      command timeout is enforced with a `timeout` wrapper inside the sandbox,
      but the HTTP request carrying it is bounded by the client transport's
      read timeout (~300 seconds on the SDK default). A longer-running command
      fails the request first; `execute` then returns an error
      `ExecuteResponse` (it does not raise), and the command may keep running
      server-side. To use timeouts above ~300 seconds, construct the
      `SandboxClient` with a transport whose read timeout exceeds them.
    - **`write` refuses to overwrite an existing file**, matching the shared
      `langchain-tests` sandbox suite (and the sibling Azure Blob backend)
      rather than the deepagents middleware's "replaces it entirely" tool
      description. The error tells the model to use `edit` or delete first.
    """

    def __init__(
        self,
        client: SandboxClient,
        *,
        async_client: AsyncSandboxClient | None = None,
        default_timeout: int = _DEFAULT_TIMEOUT,
        enable_capture_offload: bool = False,
    ) -> None:
        """Create a backend wrapping an existing ACA sandbox client.

        Args:
            client: Data-plane client for the sandbox to operate on.
            async_client: Optional async client for the *same* sandbox. When
                given, async operations use the SDK's async transport; without
                it they fall back to running the sync client in a worker
                thread. Either way the async file operations take the same SDK
                path as their sync counterparts.
            default_timeout: Seconds allowed for a command when `execute` is
                called without an explicit timeout.
            enable_capture_offload: Opt in to capture-at-source offload of
                large command output. Requires a POSIX shell and coreutils in
                the disk image -- true for the `ubuntu`, `debian`, and `python`
                presets, but not guaranteed for `alpine` or BYO OCI images,
                so it is off by default.
        """
        self._client = client
        self._async_client = async_client
        self._default_timeout = default_timeout
        self.enable_capture_offload = enable_capture_offload

    @property
    def id(self) -> str:
        """Return the sandbox id."""
        return self._client.sandbox_id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum seconds to allow the command to run. If `None`,
                uses the backend's default timeout.

        Returns:
            `ExecuteResponse` containing output, exit code, and truncation
            flag. A transport-level failure (most commonly a command outrunning
            the client's ~300 s read timeout -- see the class docstring)
            returns an error `ExecuteResponse` with `exit_code=None` rather
            than raising into the agent loop.
        """
        effective = timeout if timeout is not None else self._default_timeout
        try:
            result = self._client.exec(with_timeout(command, effective))
        except AzureError as e:
            logger.warning("ACA sandbox exec failed: %s", e)
            return execute_error_response(e)
        return execute_response(result, timeout_wrapped=effective > 0)

    async def aexecute(  # noqa: ASYNC109 - protocol-defined parameter
        self, command: str, *, timeout: int | None = None
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Uses the SDK's async client when one was supplied, otherwise defers to
        the protocol default, which offloads the blocking `execute()` to a
        worker thread.

        Args:
            command: Shell command string to execute.
            timeout: Maximum seconds to allow the command to run. If `None`,
                uses the backend's default timeout.

        Returns:
            `ExecuteResponse` containing output, exit code, and truncation
            flag. Transport-level failures return an error `ExecuteResponse`
            rather than raising, exactly as in `execute`.
        """
        if self._async_client is None:
            return await super().aexecute(command, timeout=timeout)
        effective = timeout if timeout is not None else self._default_timeout
        try:
            result = await self._async_client.exec(with_timeout(command, effective))
        except AzureError as e:
            logger.warning("ACA sandbox exec failed: %s", e)
            return execute_error_response(e)
        return execute_response(result, timeout_wrapped=effective > 0)

    async def aclose(self) -> None:
        """Close a supplied `async_client`'s connection pool.

        The narrow exception to "the caller keeps ownership": an aio client
        holds an aiohttp session that must be closed from a running event
        loop, and the code that has one at teardown time is the code calling
        this backend. Callers that prefer to keep ownership can simply close
        the client themselves and skip `aclose()` -- calling both is safe.
        No-op when no async client was supplied. There is no sync `close()`
        because the sync client's pool does not require loop-bound teardown.
        """
        if self._async_client is not None:
            await self._async_client.close()

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write content via the SDK rather than through a shell command.

        `BaseSandbox.write()` runs a `python3 -c` mkdir preflight and then
        delegates content to `upload_files` -- already this class's SDK path.
        Overriding skips the shell preflight (`create_dirs=True` covers it)
        and adds the overwrite refusal below.

        Refuses to overwrite: this matches the shared `langchain-tests`
        sandbox suite (which requires an existing path to fail untouched),
        not the deepagents middleware's tool description -- see the class
        docstring. As in the base class, there is a TOCTOU window between the
        `stat_file` check and the write.

        `AzureError` rather than `HttpResponseError`, matching the batch
        methods: a transport blip (`ServiceRequestError`) is not an HTTP
        error and would otherwise escape as an exception.

        Args:
            file_path: Destination path inside the sandbox.
            content: Text content to write.

        Returns:
            `WriteResult` with the written path, or an error message.
        """
        try:
            existing = self._client.stat_file(file_path)
        except ResourceNotFoundError:
            pass
        except AzureError as e:
            return write_error(file_path, e)
        else:
            return WriteResult(error=_exists_error(file_path, existing))

        try:
            self._client.write_file(file_path, content, create_dirs=True)
        except AzureError as e:
            return write_error(file_path, e)
        return WriteResult(path=file_path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async counterpart of `write`, on the same SDK path.

        Overridden for the same reason as `aread`.

        Args:
            file_path: Destination path inside the sandbox.
            content: Text content to write.

        Returns:
            `WriteResult` with the written path, or an error message.
        """
        client = self._async_client
        try:
            if client is None:
                existing = await asyncio.to_thread(self._client.stat_file, file_path)
            else:
                existing = await client.stat_file(file_path)
        except ResourceNotFoundError:
            pass
        except AzureError as e:
            return write_error(file_path, e)
        else:
            return WriteResult(error=_exists_error(file_path, existing))

        try:
            if client is None:
                await asyncio.to_thread(
                    partial(self._client.write_file, create_dirs=True),
                    file_path,
                    content,
                )
            else:
                await client.write_file(file_path, content, create_dirs=True)
        except AzureError as e:
            return write_error(file_path, e)
        return WriteResult(path=file_path)

    def _route_read(self, file_path: str, info: FileInfo) -> ReadResult | None:
        """Decide a read's route from its stat, before any download.

        Returns an error `ReadResult` to short-circuit, the `_PAGINATE`
        sentinel to fall back to the base class's server-side pagination, or
        `None` for the SDK download path.
        """
        if info.is_directory:
            return ReadResult(error=f"File '{file_path}': {IS_DIRECTORY}")
        size = info.size or 0
        if not is_text_file(file_path) and size > MAX_BINARY_BYTES:
            return binary_too_large_error(file_path)
        if size > _READ_DOWNLOAD_CAP:
            return _PAGINATE
        return None

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read file content via the SDK rather than through a shell command.

        `BaseSandbox.read()` pipes content through `execute()`; the SDK path
        fetches the bytes directly and applies the base class's pagination and
        binary handling locally via `read_result_from_bytes`. Because the SDK
        read returns the whole file, a `stat_file` runs first (one extra API
        call per read): an oversized binary is refused without any download,
        and a file over ~10 MiB falls back to the base class's server-side
        pagination rather than re-downloading everything for each page. That
        fallback runs the base `python3` read script through `execute()`,
        which every preset disk image can run.

        Args:
            file_path: Absolute path to the file to read.
            offset: Number of leading text lines to skip.
            limit: Maximum number of text lines to return.

        Returns:
            `ReadResult` with `file_data` on success or `error` on failure.
        """
        try:
            info = self._client.stat_file(file_path)
        except AzureError as e:
            return read_error(file_path, e)
        routed = self._route_read(file_path, info)
        if routed is _PAGINATE:
            return super().read(file_path, offset, limit)
        if routed is not None:
            return routed
        try:
            raw = self._client.read_file(file_path)
        except AzureError as e:
            return read_error(file_path, e)
        return read_result_from_bytes(file_path, raw, offset=offset, limit=limit)

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        """Async counterpart of `read`, on the same stat-then-download path.

        Overridden so async callers do not fall back to the base class's shell
        implementation.

        Args:
            file_path: Absolute path to the file to read.
            offset: Number of leading text lines to skip.
            limit: Maximum number of text lines to return.

        Returns:
            `ReadResult` with `file_data` on success or `error` on failure.
        """
        client = self._async_client
        try:
            if client is None:
                info = await asyncio.to_thread(self._client.stat_file, file_path)
            else:
                info = await client.stat_file(file_path)
        except AzureError as e:
            return read_error(file_path, e)
        routed = self._route_read(file_path, info)
        if routed is _PAGINATE:
            return await super().aread(file_path, offset, limit)
        if routed is not None:
            return routed
        try:
            if client is None:
                raw = await asyncio.to_thread(self._client.read_file, file_path)
            else:
                raw = await client.read_file(file_path)
        except AzureError as e:
            return read_error(file_path, e)
        return read_result_from_bytes(file_path, raw, offset=offset, limit=limit)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files into the sandbox.

        Supports partial success: one failure does not affect the others. Catch
        `AzureError`, not just `HttpResponseError` -- transport failures
        (`ServiceRequestError`) are not `HttpResponseError`, and letting one
        escape would abort the whole batch.

        Args:
            files: List of `(path, content)` tuples to upload.

        Returns:
            One `FileUploadResponse` per input, in the same order.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error=INVALID_PATH))
                continue
            try:
                self._client.write_file(path, content, create_dirs=True)
            except AzureError as e:
                # `warning`, matching the read/write failure paths: a failed
                # transfer is actionable operator signal, not trace detail.
                logger.warning("Failed to upload %s: %s", path, e)
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error=transfer_error(e, missing_is_file=False),
                    )
                )
            else:
                responses.append(FileUploadResponse(path=path, error=None))
        return responses

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Async counterpart of `upload_files`.

        Only overridden when an async client was supplied; otherwise the base
        class's thread offload already runs the sync SDK path above.

        Args:
            files: List of `(path, content)` tuples to upload.

        Returns:
            One `FileUploadResponse` per input, in the same order.
        """
        client = self._async_client
        if client is None:
            return await super().aupload_files(files)
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error=INVALID_PATH))
                continue
            try:
                await client.write_file(path, content, create_dirs=True)
            except AzureError as e:
                # `warning`, matching the read/write failure paths: a failed
                # transfer is actionable operator signal, not trace detail.
                logger.warning("Failed to upload %s: %s", path, e)
                responses.append(
                    FileUploadResponse(
                        path=path,
                        error=transfer_error(e, missing_is_file=False),
                    )
                )
            else:
                responses.append(FileUploadResponse(path=path, error=None))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        Supports partial success: one failure does not affect the others. See
        `upload_files` on why this catches `AzureError` rather than
        `HttpResponseError`.

        Args:
            paths: List of absolute file paths to download.

        Returns:
            One `FileDownloadResponse` per input, in the same order.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=INVALID_PATH)
                )
                continue
            try:
                content = self._client.read_file(path)
            except AzureError as e:
                logger.warning("Failed to download %s: %s", path, e)
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error=transfer_error(e)
                    )
                )
            else:
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async counterpart of `download_files`.

        See `aupload_files` on why this only overrides when an async client
        was supplied.

        Args:
            paths: List of absolute file paths to download.

        Returns:
            One `FileDownloadResponse` per input, in the same order.
        """
        client = self._async_client
        if client is None:
            return await super().adownload_files(paths)
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=INVALID_PATH)
                )
                continue
            try:
                content = await client.read_file(path)
            except AzureError as e:
                logger.warning("Failed to download %s: %s", path, e)
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error=transfer_error(e)
                    )
                )
            else:
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
        return responses
