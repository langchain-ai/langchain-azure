"""Fields and session lifecycle shared by the dynamic-sessions tools.

`SessionsPythonREPLTool` and `SessionsBashTool` talk to different endpoints
with different request bodies and response shapes, but their configuration and
session lifecycle are identical. Those live here so the two cannot drift on
cleanup or file-handle ownership, which they previously did.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import ExitStack, contextmanager
from typing import (
    TYPE_CHECKING,
    BinaryIO,
    Callable,
    ClassVar,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)
from uuid import uuid4

import requests
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

from langchain_azure_container_apps.dynamic_sessions._common import (
    REQUEST_TIMEOUT,
    access_token_provider_factory,
    auth_headers,
    build_session_url,
)

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

SESSION_DELETE_API_VERSION = "2025-10-02-preview"

_ToolT = TypeVar("_ToolT", bound="_BaseSessionsTool")


def _delete_session(
    *,
    pool_management_endpoint: str,
    session_id: str,
    access_token: Optional[str],
    timeout: int = REQUEST_TIMEOUT,
) -> None:
    """Delete a session from the pool."""
    api_url = build_session_url(
        pool_management_endpoint,
        "session",
        session_id=session_id,
        api_version=SESSION_DELETE_API_VERSION,
    )
    response = requests.delete(
        api_url, headers=auth_headers(access_token), timeout=timeout
    )
    response.raise_for_status()


class _BaseSessionsTool(BaseTool):
    """Configuration and session lifecycle common to the sessions tools."""

    _API_VERSION: ClassVar[str] = ""
    """Data-plane API version. The two tools are pinned to different ones."""

    pool_management_endpoint: str
    """The management endpoint of the session pool."""

    # Class-level default, deliberately: every non-overriding tool instance
    # shares one provider and therefore one cached token. The token's scope is
    # global to the service (not per pool or session), so sharing is safe and
    # avoids a credential round-trip per instance. Do not "fix" this into a
    # default_factory.
    access_token_provider: Callable[[], Optional[str]] = access_token_provider_factory()
    """A function that returns the access token to use for the session pool.

    Returning ``None`` (or an empty string) is an error at request time: every
    call needs a bearer token, so a missing one raises ``ValueError`` rather
    than sending an unauthenticated request.
    """

    request_timeout: int = Field(default=REQUEST_TIMEOUT, gt=0)
    """Seconds Requests waits for each HTTP call (default 120).

    Executions are synchronous server-side -- the response arrives when the
    code finishes -- so code that legitimately runs longer than this needs a
    larger value. The old tools sent no timeout at all and could hang forever
    on a wedged connection.

    Rejected at construction rather than at the first request: Requests raises
    ``ValueError`` on a zero or negative timeout, which would otherwise surface
    from an unrelated-looking call much later. There is deliberately no
    "wait forever" spelling here; the backend's separate ``timeout=0`` handling
    exists only because the Deep Agents protocol defines 0 as "no timeout".
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    """The session ID to use. Defaults to a random UUID."""

    delete_session_after_invocation: bool = False
    """Whether to delete the session after each tool invocation.

    When ``True``, the session is deleted after each call to ``run``/``invoke``.
    When ``False`` (default), the same session ID is reused across invocations.
    """

    sanitize_input: bool = True
    """Whether to strip code fences and interpreter names from the input."""

    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    _session_id_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _get_session_id(self) -> str:
        with self._session_id_lock:
            return self.session_id

    def _build_url(
        self, path: str, *, params: Optional[Mapping[str, str]] = None
    ) -> str:
        return build_session_url(
            self.pool_management_endpoint,
            path,
            session_id=self._get_session_id(),
            api_version=self._API_VERSION,
            params=params,
        )

    def _auth_headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers = auth_headers(self.access_token_provider())
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    @contextmanager
    def _upload_payload(
        self,
        *,
        data: Optional[BinaryIO],
        local_file_path: Optional[str],
        remote_file_path: Optional[str],
    ) -> Iterator[Tuple[str, BinaryIO]]:
        """Yield the ``(remote name, stream)`` to send for an upload.

        Only a handle opened here is closed; a caller-supplied `data` stream
        belongs to the caller and must outlive this call.

        Raises:
            ValueError: If neither or both of `data` and `local_file_path` are
                given, or if `data` is given without `remote_file_path` --
                there is no filename to default to, and an `assert` here would
                vanish under ``python -O``.
        """
        if data and local_file_path:
            raise ValueError("data and local_file_path cannot be provided together")
        if not data and not local_file_path:
            raise ValueError("data or local_file_path must be provided")
        if data and not remote_file_path:
            raise ValueError("remote_file_path is required when data is provided")

        with ExitStack() as stack:
            if data:
                stream = data
            else:
                assert local_file_path is not None
                if not remote_file_path:
                    remote_file_path = os.path.basename(local_file_path)
                stream = stack.enter_context(open(local_file_path, "rb"))
            assert remote_file_path is not None
            yield remote_file_path, stream

    def _delete_session_sync(self, session_id: str) -> None:
        _delete_session(
            pool_management_endpoint=self.pool_management_endpoint,
            session_id=session_id,
            access_token=self.access_token_provider(),
            timeout=self.request_timeout,
        )

    def _delete_session_in_background(self, session_id: str) -> None:
        """Fire the session DELETE from a daemon thread.

        Thread-based, not asyncio. Being a daemon thread, it can be killed at
        interpreter exit before the DELETE lands; the session then lingers
        until the pool reclaims it on cooldown -- an accepted trade for never
        blocking or crashing the caller on cleanup.
        """

        def _worker() -> None:
            try:
                self._delete_session_sync(session_id)
            except Exception:
                logger.warning("Failed to delete session %s", session_id, exc_info=True)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def delete_session(self) -> None:
        """Delete current session in the background and rotate session ID."""
        with self._session_id_lock:
            session_id_to_delete = self.session_id
            self.session_id = str(uuid4())
        self._delete_session_in_background(session_id_to_delete)

    def close(self) -> None:
        """Close the current session by deleting it from the pool."""
        self.delete_session()

    def __enter__(self: _ToolT) -> _ToolT:
        """Enter context manager scope."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit context manager scope and close the session."""
        self.close()
