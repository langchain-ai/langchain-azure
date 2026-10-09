"""Internal state and strict checkpoint access for response-ID branches."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from azure.ai.agentserver.core import (
    AgentConfig,
    FoundryAgentRequestContext,
    reset_request_context,
    set_request_context,
)
from azure.ai.agentserver.core.storage import (
    FoundryStateStore,
    FoundryStorageConflictError,
    FoundryStorageNotFoundError,
    FoundryStoragePreconditionError,
)
from azure.ai.agentserver.responses import (
    FoundryResourceNotFoundError,
    PlatformContext,
    ResponseContext,
    ResponseEventStream,
    ResponseProviderProtocol,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .._response_instructions import _INSTRUCTIONS_MODE_HEADER, _InstructionProvenance
from .checkpoint_ref import CheckpointRef
from .conversation_chain_store import ConversationChainStoreProtocol
from .task_storage_manager import TaskStorageManager

BRANCH_MODE_HEADER = "x-client-langchain-response-branching"
BRANCH_OWNER_HEADER = "x-client-langchain-response-owner"
BRANCH_MODE = "checkpoint-v1"
BRANCH_ORIGIN_KEY = "langgraph_branch_origin_v1"
BRANCH_BOUNDARY_KEY = "langgraph_response_boundary_v1"
BRANCH_MODE_METADATA = "langgraph_response_branching"
logger = logging.getLogger(__name__)


@dataclass
class _ResponseAdmission:
    request: Request
    owner: str
    started: bool = False


_response_admission: ContextVar[_ResponseAdmission | None] = ContextVar(
    "response_admission", default=None
)


def mark_response_started() -> None:
    """Prevent rollback once the SDK invokes the registered response handler."""
    admission = _response_admission.get()
    if admission is not None:
        admission.started = True


class BranchingAdmissionMiddleware:
    """Validate public linkage and stamp trusted mode into persisted headers.

    Args:
        app: The next ASGI application.
        enabled: Whether fresh requests may use response branching. Incoming
            client values are removed even when the feature is disabled.
        executions: Atomic ownership records for admitted executions.
        provider: The configured Responses provider used to reject reused IDs.
        instructions_mode: Instruction transport for fresh requests. Defaults
            to the existing message-based behavior.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        enabled: bool,
        executions: ResponseExecutionStore | None = None,
        provider: ResponseProviderProtocol | None = None,
        instructions_mode: Literal["messages", "context"] = "messages",
    ) -> None:
        self.app = app
        self.enabled = enabled
        self.executions = executions
        self.provider = provider
        self.instructions_mode = instructions_mode

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject invalid create requests before SDK admission or execution."""
        admission = None
        if scope["type"] == "http":
            internal_headers = {
                BRANCH_MODE_HEADER.encode("ascii"),
                BRANCH_OWNER_HEADER.encode("ascii"),
                _INSTRUCTIONS_MODE_HEADER.encode("ascii"),
            }
            headers = [
                (name, value)
                for name, value in scope.get("headers", [])
                if name.lower() not in internal_headers
            ]
            headers.append(
                (
                    _INSTRUCTIONS_MODE_HEADER.encode("ascii"),
                    self.instructions_mode.encode("ascii"),
                )
            )
            if self.enabled:
                headers.append(
                    (BRANCH_MODE_HEADER.encode("ascii"), BRANCH_MODE.encode("ascii"))
                )
            scope = {**scope, "headers": headers}
            if scope["method"] == "POST" and scope["path"].rstrip("/").endswith(
                "/responses"
            ):
                request = Request(scope, receive)
                body = await request.body()
                try:
                    payload = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    payload = None
                if isinstance(payload, dict):
                    invalid = self._invalid_linkage(payload)
                    if invalid is not None:
                        parameter, message = invalid
                        await JSONResponse(
                            {
                                "error": {
                                    "type": "invalid_request_error",
                                    "message": message,
                                    "param": parameter,
                                    "code": None,
                                }
                            },
                            status_code=400,
                        )(scope, receive, send)
                        return
                    if self.enabled and payload.get("conversation") is None:
                        owner = uuid.uuid4().hex
                        failure = await self._admit(request, owner)
                        if failure is not None:
                            await failure(scope, receive, send)
                            return
                        headers.append(
                            (BRANCH_OWNER_HEADER.encode("ascii"), owner.encode("ascii"))
                        )
                        scope = {**scope, "headers": headers}
                        admission = _ResponseAdmission(request, owner)
                original_receive = receive
                body_sent = False

                async def replay_body() -> Any:
                    nonlocal body_sent
                    if not body_sent:
                        body_sent = True
                        return {"type": "http.request", "body": body}
                    return await original_receive()

                receive = replay_body
        if admission is None:
            await self.app(scope, receive, send)
            return

        error_types = {400: "invalid_request_error", 404: "not_found_error"}
        rejection_type = None
        buffered: list[Message] = []
        response_body = bytearray()

        async def observe_send(message: Message) -> None:
            nonlocal rejection_type
            if message["type"] == "http.response.start":
                rejection_type = error_types.get(message["status"])
            if rejection_type is not None:
                buffered.append(message)
                if message["type"] == "http.response.body":
                    response_body.extend(message.get("body", b""))
            else:
                await send(message)

        token = _response_admission.set(admission)
        try:
            await self.app(scope, receive, observe_send)
            if (
                buffered
                and buffered[-1]["type"] == "http.response.body"
                and not buffered[-1].get("more_body", False)
                and not admission.started
            ):
                try:
                    result = json.loads(response_body)
                except (ValueError, UnicodeDecodeError):
                    result = None
                error = result.get("error") if isinstance(result, dict) else None
                if isinstance(error, dict) and error.get("type") == rejection_type:
                    await self._release_rejected(admission)
            for response_message in buffered:
                await send(response_message)
        finally:
            _response_admission.reset(token)

    async def _admit(self, request: Request, owner: str) -> JSONResponse | None:
        response_id = request.headers.get("x-agent-response-id", "").strip()
        if not response_id:
            return None
        platform = PlatformContext(
            user_id_key=request.headers.get("x-agent-user-id"),
            call_id=request.headers.get("x-agent-foundry-call-id"),
        )
        token = set_request_context(
            FoundryAgentRequestContext(
                user_id=platform.user_id_key, call_id=platform.call_id
            )
        )
        try:
            if self.executions is None or self.provider is None:
                raise RuntimeError("Response admission is not configured.")
            identity = self.executions.response_identity(response_id, platform)
            try:
                await self.provider.get_response(response_id, context=platform)
            except (KeyError, FoundryResourceNotFoundError):
                if await self.executions.claim("response", identity, owner):
                    return None
            return JSONResponse(
                {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "This response identity has already been admitted.",
                        "param": None,
                        "code": None,
                    }
                },
                status_code=409,
            )
        except Exception:
            logger.exception("Response admission failed")
            return JSONResponse(
                {
                    "error": {
                        "type": "server_error",
                        "message": "The response could not be admitted.",
                        "param": None,
                        "code": None,
                    }
                },
                status_code=500,
            )
        finally:
            reset_request_context(token)

    async def _release_rejected(self, admission: _ResponseAdmission) -> None:
        response_id = admission.request.headers.get("x-agent-response-id", "").strip()
        if not response_id or self.executions is None or self.provider is None:
            return
        platform = PlatformContext(
            user_id_key=admission.request.headers.get("x-agent-user-id"),
            call_id=admission.request.headers.get("x-agent-foundry-call-id"),
        )
        token = set_request_context(
            FoundryAgentRequestContext(
                user_id=platform.user_id_key, call_id=platform.call_id
            )
        )
        try:
            try:
                await self.provider.get_response(response_id, context=platform)
            except (KeyError, FoundryResourceNotFoundError):
                if not admission.started:
                    identity = self.executions.response_identity(response_id, platform)
                    await self.executions.release("response", identity, admission.owner)
        except Exception:
            logger.exception("Failed to release a rejected response identity")
        finally:
            reset_request_context(token)

    @staticmethod
    def _invalid_linkage(payload: dict[str, Any]) -> tuple[str, str] | None:
        if "response_id" in payload:
            return "response_id", "Unknown parameter: response_id."
        parent = payload.get("previous_response_id")
        if parent is not None:
            if not isinstance(parent, str) or not parent.strip():
                return (
                    "previous_response_id",
                    "previous_response_id must be a non-empty string or null.",
                )
            if payload.get("conversation") is not None:
                return (
                    "previous_response_id",
                    "previous_response_id and conversation are mutually exclusive.",
                )
        return None


class BranchingError(ValueError):
    """A branch failure with an internal reason code and client-safe message."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ResponseExecutionStore:
    """Persist atomic execution ownership independently of mutable pointers."""

    def __init__(self) -> None:
        config = AgentConfig.from_env()
        deployment = json.dumps(
            [
                config.project_id,
                config.agent_id,
                config.agent_name,
                config.agent_version,
            ]
        )
        deployment_key = hashlib.sha256(deployment.encode()).hexdigest()
        self._name = (
            f"langchain_azure_ai.agents.hosting/response_owners/{deployment_key}"
        )

    @staticmethod
    def _key(kind: str, identity: str) -> str:
        return f"{kind}:{hashlib.sha256(identity.encode()).hexdigest()}"

    @staticmethod
    def response_identity(response_id: str, platform: PlatformContext) -> str:
        """Scope an opaque response identity to the platform's caller partition."""
        return json.dumps([platform.user_id_key, response_id])

    async def owner(self, kind: str, identity: str) -> str | None:
        """Read an ownership tombstone, propagating storage or data errors."""
        state_store = await FoundryStateStore.get_or_create(
            self._name, item_ttl_seconds=-1
        )
        async with state_store:
            item = await state_store.get_item(self._key(kind, identity))
        if item is None:
            return None
        if (
            not isinstance(item.value, dict)
            or item.value.get("version") != "1"
            or not isinstance(item.value.get("owner"), str)
            or not item.value["owner"]
        ):
            raise BranchingError(
                "invalid_branch_state", "The execution ownership record is invalid."
            )
        return item.value["owner"]

    async def claim(self, kind: str, identity: str, owner: str) -> bool:
        """Claim once, allowing only the same admitted execution to re-enter."""
        state_store = await FoundryStateStore.get_or_create(
            self._name, item_ttl_seconds=-1
        )
        async with state_store:
            try:
                await state_store.create_item(
                    self._key(kind, identity), {"version": "1", "owner": owner}
                )
            except FoundryStorageConflictError:
                return await self.owner(kind, identity) == owner
        return True

    async def release(self, kind: str, identity: str, owner: str) -> bool:
        """Release an unadmitted claim only if its owner and version still match."""
        state_store = await FoundryStateStore.get_or_create(
            self._name, item_ttl_seconds=-1
        )
        async with state_store:
            key = self._key(kind, identity)
            item = await state_store.get_item(key)
            if item is None or item.value != {"version": "1", "owner": owner}:
                return False
            if not isinstance(item.etag, str) or not item.etag or item.etag == "*":
                raise BranchingError(
                    "invalid_branch_state",
                    "The execution ownership version is invalid.",
                )
            try:
                await state_store.delete_item(key, if_match=item.etag)
            except (FoundryStorageNotFoundError, FoundryStoragePreconditionError):
                return False
        return True


class ResponseBranchStore:
    """Keep confirmed origins separate from completed response boundaries.

    Args:
        store: The existing conversation-chain store. Execution ownership is
            managed separately through atomic claims.
        provider: The configured response provider for authorized parent lookup.
    """

    def __init__(
        self,
        store: ConversationChainStoreProtocol,
        provider: ResponseProviderProtocol | None = None,
    ) -> None:
        self._store = store
        self._provider = provider

    @staticmethod
    def _record(
        ref: CheckpointRef, *, paused: bool, pause_id: str = ""
    ) -> dict[str, str]:
        return {
            "version": "1",
            "checkpoint_ns": "",
            "paused": str(paused).lower(),
            "pause_id": pause_id,
            **ref.to_dict(),
        }

    @staticmethod
    def _reference(record: Any) -> CheckpointRef:
        if (
            not isinstance(record, dict)
            or record.get("version") != "1"
            or record.get("checkpoint_ns") != ""
            or record.get("paused") not in {"true", "false"}
            or (
                record.get("paused") == "true"
                and (
                    not isinstance(record.get("pause_id"), str)
                    or not record["pause_id"]
                )
            )
        ):
            raise BranchingError(
                "invalid_branch_state", "The response checkpoint record is invalid."
            )
        ref = CheckpointRef.from_dict(record)
        if ref is None:
            raise BranchingError(
                "checkpoint_unavailable", "The response checkpoint is unavailable."
            )
        return ref

    async def prepare(
        self,
        *,
        response_key: str,
        parent_key: str,
        parent_id: str,
        context: ResponseContext,
    ) -> CheckpointRef:
        """Confirm the parent origin or restore an unpublished execution."""
        existing = await self._store.get(response_key, BRANCH_ORIGIN_KEY)
        if context.is_recovery:
            if await self._store.get(response_key, BRANCH_BOUNDARY_KEY) is not None:
                raise BranchingError(
                    "invalid_branch_state",
                    "A response with a published boundary cannot be resumed.",
                )
            ref = self._reference(existing)
            if (
                existing is None
                or existing.get("mode") != BRANCH_MODE
                or existing.get("parent_response_id") != parent_id
            ):
                raise BranchingError(
                    "invalid_branch_state", "The confirmed branch origin is invalid."
                )
            return ref

        provider = self._provider
        if provider is None:
            raise BranchingError(
                "invalid_branch_state", "The Responses provider is unavailable."
            )
        parent = await provider.get_response(
            parent_id, context=context.platform_context
        )
        if parent is None or parent.get("status") != "completed":
            raise BranchingError(
                "invalid_branch_state",
                "The parent response must be stored and completed.",
            )
        metadata = (parent.get("metadata") or {}).get("_internal_metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as exc:
                raise BranchingError(
                    "invalid_branch_state", "The parent checkpoint metadata is invalid."
                ) from exc
        if not isinstance(metadata, Mapping):
            raise BranchingError(
                "invalid_branch_state", "The parent checkpoint metadata is invalid."
            )
        boundary = metadata.get(BRANCH_BOUNDARY_KEY)
        if boundary is not None:
            if (
                isinstance(boundary, dict)
                and "thread_id" not in boundary
                and "checkpoint_id" not in boundary
            ):
                captured_ref = TaskStorageManager(dict(metadata)).checkpoint_ref
                if captured_ref is None:
                    raise BranchingError(
                        "checkpoint_unavailable",
                        "The parent has no completed checkpoint.",
                    )
                boundary = {**boundary, **captured_ref.to_dict()}
            ref = self._reference(boundary)
            indexed = await self._store.get(parent_key, BRANCH_BOUNDARY_KEY)
            if indexed != boundary:
                raise BranchingError(
                    "invalid_branch_state",
                    "The parent checkpoint records do not agree.",
                )
        else:
            if BRANCH_MODE_METADATA in metadata:
                raise BranchingError(
                    "checkpoint_unavailable", "The parent has no completed checkpoint."
                )
            raise BranchingError(
                "invalid_branch_state",
                "The parent response was not created with response branching.",
            )

        origin = {
            **boundary,
            "mode": BRANCH_MODE,
            "parent_response_id": parent_id,
        }
        if existing is not None and existing != origin:
            raise BranchingError(
                "invalid_branch_state", "The confirmed branch origin cannot be changed."
            )
        if existing is None:
            await self._store.set(response_key, BRANCH_ORIGIN_KEY, origin)
        return ref

    async def check_pause_owner(
        self,
        response_key: str,
        executions: ResponseExecutionStore,
        *,
        claim: bool = False,
    ) -> None:
        """Protect a paused origin across all response aliases and recoveries."""
        origin = await self._store.get(response_key, BRANCH_ORIGIN_KEY)
        ref = self._reference(origin)
        if origin is None or origin["paused"] != "true":
            if claim:
                raise BranchingError(
                    "unsupported_approval_branch",
                    "The legacy checkpoint has no verifiable pause ownership.",
                )
            return
        identity = json.dumps(
            {**ref.to_dict(), "pause_id": origin["pause_id"]}, sort_keys=True
        )
        if claim:
            available = await executions.claim("pause", identity, response_key)
        else:
            owner = await executions.owner("pause", identity)
            available = owner is None or owner == response_key
        if not available:
            raise BranchingError(
                "unsupported_approval_branch",
                "This paused checkpoint already has a continuation. "
                "Independent historical approval branches are not supported.",
            )

    async def publish(
        self,
        response_key: str,
        stream: ResponseEventStream,
        ref: CheckpointRef | None,
        *,
        paused: bool,
        continue_pause: bool = False,
    ) -> None:
        """Index this run's checkpoint before the SDK commits its terminal event."""
        if ref is None:
            raise BranchingError(
                "checkpoint_unavailable",
                "The response produced no checkpoint boundary.",
            )
        pause_id = hashlib.sha256(response_key.encode()).hexdigest() if paused else ""
        if continue_pause:
            origin = await self._store.get(response_key, BRANCH_ORIGIN_KEY)
            origin_ref = self._reference(origin)
            if origin is None or origin_ref != ref or origin["paused"] != "true":
                raise BranchingError(
                    "invalid_branch_state", "The pending pause origin is invalid."
                )
            pause_id = origin["pause_id"]
        record = self._record(ref, paused=paused, pause_id=pause_id)
        existing = await self._store.get(response_key, BRANCH_BOUNDARY_KEY)
        if existing is not None and existing != record:
            raise BranchingError(
                "invalid_branch_state", "The response boundary cannot be replaced."
            )
        if TaskStorageManager.from_stream(stream).checkpoint_ref != ref:
            raise BranchingError(
                "invalid_branch_state",
                "The completed boundary does not match the execution checkpoint.",
            )
        stream.internal_metadata[BRANCH_BOUNDARY_KEY] = {
            key: value
            for key, value in record.items()
            if key not in {"thread_id", "checkpoint_id"}
        }
        if existing is None:
            await self._store.set(response_key, BRANCH_BOUNDARY_KEY, record)


class ResponseCheckpointSaver(BaseCheckpointSaver[Any]):
    """Track host instructions and enforce exact checkpoint reads for branches.

    Host-owned instruction metadata is cleared only after verified removal
    writes. Pending writes restored from the saver preserve that evidence
    across adapter recreation. Messages and parent checkpoints are unchanged.

    Args:
        saver: The graph-owned saver. Ownership and lifecycle stay with its
            caller; this request-scoped adapter does not open or close it.
        branching: Whether this request uses response branching. Branching
            rejects unavailable or mismatched explicit checkpoints; otherwise
            reads preserve the wrapped saver's behavior. Both modes propagate
            backend errors and track verified instruction deletions.

    Attributes:
        on_next_load: Optional hook awaited after the next asynchronous load
            passes validation, before returning the checkpoint to LangGraph.
            Cleared after success; used to defer execution ownership claims.
    """

    def __init__(self, saver: BaseCheckpointSaver[Any], *, branching: bool) -> None:
        super().__init__(serde=saver.serde)
        self._saver = saver
        self._branching = branching
        self._instructions = _InstructionProvenance()
        self.on_next_load: Callable[[], Awaitable[None]] | None = None

    @property
    def config_specs(self) -> Any:
        """Preserve the wrapped saver's configurable fields."""
        return self._saver.config_specs

    def _validate(
        self, config: RunnableConfig, saved: CheckpointTuple | None
    ) -> CheckpointTuple | None:
        requested = config.get("configurable") or {}
        checkpoint_id = requested.get("checkpoint_id")
        if not self._branching or not checkpoint_id:
            return saved
        if saved is None:
            raise BranchingError(
                "checkpoint_unavailable", "The required checkpoint is unavailable."
            )
        actual = saved.config.get("configurable") or {}
        if (
            actual.get("thread_id") != requested.get("thread_id")
            or actual.get("checkpoint_ns", "") != requested.get("checkpoint_ns", "")
            or actual.get("checkpoint_id") != checkpoint_id
            or saved.checkpoint.get("id") != checkpoint_id
        ):
            raise BranchingError(
                "checkpoint_unavailable",
                "The saver did not return the required checkpoint.",
            )
        return saved

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read a checkpoint and retain its instruction-removal evidence."""
        saved = self._validate(config, self._saver.get_tuple(config))
        self._instructions.observe(saved)
        return saved

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read an async checkpoint and retain instruction-removal evidence."""
        saved = self._validate(config, await self._saver.aget_tuple(config))
        self._instructions.observe(saved)
        if self.on_next_load is not None:
            await self.on_next_load()
            self.on_next_load = None
        return saved

    def list(self, *args: Any, **kwargs: Any) -> Iterator[CheckpointTuple]:
        """Delegate checkpoint history without changing retention."""
        return self._saver.list(*args, **kwargs)

    async def alist(self, *args: Any, **kwargs: Any) -> AsyncIterator[CheckpointTuple]:
        """Delegate asynchronous checkpoint history without changing retention."""
        async for saved in self._saver.alist(*args, **kwargs):
            yield saved

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Write a checkpoint using the original saver."""
        metadata = self._instructions.checkpoint_metadata(config, checkpoint, metadata)
        return self._saver.put(config, checkpoint, metadata, new_versions)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Write an asynchronous checkpoint using the original saver."""
        metadata = self._instructions.checkpoint_metadata(config, checkpoint, metadata)
        return await self._saver.aput(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Preserve the saver's pending-write semantics."""
        self._instructions.record_writes(config, writes, task_id)
        self._saver.put_writes(config, writes, task_id, task_path)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Preserve the saver's asynchronous pending-write semantics."""
        self._instructions.record_writes(config, writes, task_id)
        await self._saver.aput_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        """Delete a thread's checkpoints and writes using the original saver.

        Args:
            thread_id: Thread whose checkpoints and pending writes are deleted.
        """
        self._saver.delete_thread(thread_id)

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete a thread's checkpoints and writes using the async saver method.

        Args:
            thread_id: Thread whose checkpoints and pending writes are deleted.
        """
        await self._saver.adelete_thread(thread_id)

    def get_next_version(self, current: Any, channel: Any) -> Any:
        """Allocate versions with the original saver's version scheme."""
        return self._saver.get_next_version(current, channel)
