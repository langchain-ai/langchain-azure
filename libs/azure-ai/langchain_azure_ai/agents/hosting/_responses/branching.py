# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request admission and exact completed-parent selection for Responses."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    reset_request_context,
    set_request_context,
)
from azure.ai.agentserver.core.platform_headers import (
    ERROR_SOURCE,
    FOUNDRY_CALL_ID,
    USER_ID,
)
from azure.ai.agentserver.responses import (
    PlatformContext,
    ResponseContext,
    ResponseEventStream,
    ResponseProviderProtocol,
)
from azure.ai.agentserver.responses.hosting._validation import (
    error_response,
    invalid_request_response,
    not_found_response,
    parse_and_validate_create_response,
)
from azure.ai.agentserver.responses.models._errors import RequestValidationError
from azure.ai.agentserver.responses.models.runtime import ResponseModeFlags
from azure.ai.agentserver.responses.store._foundry_errors import (
    FoundryApiError,
    FoundryBadRequestError,
    FoundryResourceNotFoundError,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langgraph.checkpoint.base import BaseCheckpointSaver
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .hosting_runnable_config import HostingRunnableConfig, _scope_thread_id
from .task_storage_manager import (
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_THREAD_ID,
    ResponseCheckpointSaver,
    TaskStorageManager,
)

if TYPE_CHECKING:
    from azure.ai.agentserver.responses.models import CreateResponse
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import StateSnapshot

    from .checkpoint_ref import CheckpointRef

logger = logging.getLogger(__name__)

METADATA_RESPONSE_BRANCHING = "langgraph_response_branching"
BRANCHING_MODE = "checkpoint-v1"
_MODE_HEADER = "x-client-langgraph-response-branching"


def validate_branching_saver(saver: object) -> BaseCheckpointSaver:
    """Reject missing or unimplemented async saver operations at startup."""
    if not isinstance(saver, BaseCheckpointSaver):
        raise ValueError("Response branching requires a LangGraph checkpointer.")
    if any(
        getattr(saver.__class__, name, None) is getattr(BaseCheckpointSaver, name)
        for name in ("aget_tuple", "aput", "aput_writes", "alist")
    ):
        raise ValueError(
            "Response branching requires asynchronous checkpoint reads, "
            "writes, pending writes, and history."
        )
    return saver


@dataclass(frozen=True)
class BranchSelection:
    """A request-owned parent selection, never a mutable thread head."""

    checkpoint_ref: CheckpointRef | None = None
    ordinary_continuation: bool = False
    tool_call_ids: frozenset[str] = frozenset()


_SelectionKey = tuple[int, str | None, str | None, str]
_SelectionCache = dict[_SelectionKey, BranchSelection]
_selections: ContextVar[_SelectionCache | None] = ContextVar(
    "langchain_azure_branch_selection", default=None
)


def branch_selection_cache() -> _SelectionCache:
    """Reuse HTTP admission state, or create a recovery-owned request cache."""
    cache = _selections.get()
    return cache if cache is not None else {}


@contextmanager
def branch_selection_scope(
    selections: _SelectionCache | None = None,
) -> Iterator[_SelectionCache]:
    """Bind the same request cache across SDK generator-resumption tasks."""
    cache = selections if selections is not None else branch_selection_cache()
    token = _selections.set(cache)
    try:
        yield cache
    finally:
        _selections.reset(token)


def response_provider(owner: Any) -> ResponseProviderProtocol:
    """Isolate the SDK's provider access; its default is resolved by the SDK."""
    provider = getattr(owner, "_provider", None)
    if provider is None:
        provider = getattr(getattr(owner, "_orchestrator", None), "_provider", None)
    if not isinstance(provider, ResponseProviderProtocol):
        raise RuntimeError("The Responses host has no response storage provider.")
    return provider


def response_branching_enabled(context: ResponseContext, configured: bool) -> bool:
    """Restore the admitted strategy rather than applying a new flag to recovery."""
    if not context.is_recovery:
        return configured
    headers = context.client_headers
    admitted = headers.get(_MODE_HEADER) if isinstance(headers, Mapping) else None
    metadata: Mapping[str, Any] = {}
    if context.persisted_response is not None:
        metadata = ResponseEventStream(
            response_id=context.response_id, response=context.persisted_response
        ).internal_metadata
    saved = metadata.get(METADATA_RESPONSE_BRANCHING)
    if (
        admitted not in (None, "off", BRANCHING_MODE)
        or saved not in (None, BRANCHING_MODE)
        or (saved is not None and saved != admitted)
        or (
            admitted == BRANCHING_MODE
            and saved is None
            and any(
                key in metadata
                for key in (
                    METADATA_LANGGRAPH_THREAD_ID,
                    METADATA_LANGGRAPH_CHECKPOINT_ID,
                )
            )
        )
    ):
        raise RuntimeError("The response's admitted branching mode is inconsistent.")
    return admitted == BRANCHING_MODE


def _validate_replies(items: Any, output: Any) -> tuple[bool, frozenset[str]]:
    answered = {
        item.get("call_id")
        for item in output
        if item.get("type") == "function_call_output"
    }
    calls = {
        item.get("call_id")
        for item in output
        if item.get("type") == "function_call" and item.get("call_id") not in answered
    }
    approvals = {
        item.get("id") for item in output if item.get("type") == "mcp_approval_request"
    }
    replies = False
    for item in items:
        kind = item.get("type")
        if kind not in {"function_call_output", "mcp_approval_response"}:
            continue
        key, expected = (
            ("call_id", calls)
            if kind == "function_call_output"
            else ("approval_request_id", approvals)
        )
        if item.get(key) not in expected:
            raise RequestValidationError(
                "The reply does not match an outstanding call "
                "in the previous response.",
                param="input",
            )
        replies = True
    if not replies and (calls or approvals):
        raise RequestValidationError(
            "The previous response is awaiting tool results or approval.",
            param="previous_response_id",
        )
    return replies, frozenset(calls)


async def validate_branch_checkpoint(
    graph: CompiledStateGraph, config: RunnableConfig
) -> StateSnapshot:
    """Require a restorable exact root boundary, not a paused or unfinished graph."""
    state = await graph.aget_state(config)
    if (
        HostingRunnableConfig(state.config).checkpoint_ref
        != HostingRunnableConfig(config).checkpoint_ref
        or (state.config.get("configurable") or {}).get("checkpoint_ns", "") != ""
        or state.created_at is None
    ):
        raise RuntimeError("The parent's graph checkpoint could not be restored.")
    if state.next or state.tasks or state.interrupts:
        raise RequestValidationError(
            "The previous response's graph has not finished.",
            param="previous_response_id",
        )
    return state


async def select_branch_checkpoint(
    request: CreateResponse,
    context: ResponseContext,
    graph: CompiledStateGraph,
) -> BranchSelection:
    """Resolve and validate one authorized parent, once per request."""
    parent_id = request.get("previous_response_id")
    if not isinstance(parent_id, str) or not parent_id:
        raise RequestValidationError(
            "previous_response_id must be a nonempty string.",
            param="previous_response_id",
        )
    provider = response_provider(context)
    platform = context.platform_context
    key = (id(provider), platform.user_id_key, platform.call_id, parent_id)
    cache = _selections.get()
    if cache is not None and key in cache:
        return cache[key]
    try:
        parent = await provider.get_response(parent_id, context=platform)
    except KeyError as exc:
        raise RequestValidationError(
            "The previous response was not found or was not retained.",
            code="previous_response_not_found",
            param="previous_response_id",
        ) from exc
    if parent.get("id") != parent_id:
        raise RuntimeError("The response provider returned a different parent.")
    storage = TaskStorageManager.from_stream(
        ResponseEventStream(response_id=parent_id, response=parent)
    )
    reference = storage.checkpoint_ref
    if reference is not None:
        namespace = _scope_thread_id("", context)
        if (namespace and not reference.thread_id.startswith(namespace)) or (
            not namespace and re.match(r"user-[0-9a-f]{64}:", reference.thread_id)
        ):
            raise RequestValidationError(
                "The previous response was not found or was not retained.",
                code="previous_response_not_found",
                param="previous_response_id",
            )
    if parent.get("status") != "completed":
        raise RequestValidationError(
            "The previous response must be completed.",
            param="previous_response_id",
        )
    ordinary, call_ids = _validate_replies(
        await context.get_input_items(), parent.get("output", [])
    )
    if ordinary:
        selection = BranchSelection(ordinary_continuation=True, tool_call_ids=call_ids)
    else:
        reference = storage.recovery_checkpoint_ref
        if reference is None:
            raise RuntimeError("The parent has no saved graph checkpoint.")
        config = merge_configs(
            graph.config,
            HostingRunnableConfig.create_from_checkpoint(
                reference, context
            ).runnable_config,
        )
        validation_graph = copy(graph)
        validation_graph.checkpointer = ResponseCheckpointSaver(
            validate_branching_saver(graph.checkpointer),
            thread_id=reference.thread_id,
            response_key=context.response_id,
            parent_response_id=parent_id,
        )
        await validate_branch_checkpoint(validation_graph, config)
        selection = BranchSelection(reference)
    if cache is not None:
        cache[key] = selection
    return selection


class ResponseBranchingMiddleware:
    """Validate before SDK execution and persist a server-controlled admission mode."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        graph: CompiledStateGraph,
        provider: ResponseProviderProtocol | None,
        enabled: bool,
        path: str,
    ) -> None:
        self.app = app
        self.graph = graph
        self.provider = provider
        self.enabled = enabled
        self.path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "").removeprefix(scope.get("root_path", ""))
        if scope["type"] != "http" or scope["method"] != "POST" or path != self.path:
            await self.app(scope, receive, send)
            return
        header = _MODE_HEADER.encode()
        scope = {
            **scope,
            "headers": [
                (key, value) for key, value in scope["headers"] if key.lower() != header
            ]
            + [(header, (BRANCHING_MODE if self.enabled else "off").encode())],
        }
        if not self.enabled:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        body = await request.body()
        platform = PlatformContext(
            user_id_key=request.headers.get(USER_ID),
            call_id=request.headers.get(FOUNDRY_CALL_ID),
        )
        token = set_request_context(
            FoundryAgentRequestContext(
                user_id=platform.user_id_key, call_id=platform.call_id
            )
        )
        try:
            with branch_selection_scope():
                error = await self._admit(body, platform)
                if error is not None:
                    await error(scope, receive, send)
                    return
                replayed = False

                async def replay_body() -> Message:
                    nonlocal replayed
                    if not replayed:
                        replayed = True
                        return {
                            "type": "http.request",
                            "body": body,
                            "more_body": False,
                        }
                    return await receive()

                await self.app(scope, replay_body, send)
        finally:
            reset_request_context(token)

    async def _admit(self, body: bytes, platform: PlatformContext) -> Response | None:
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                if "response_id" in payload:
                    raise RequestValidationError(
                        "response_id is assigned by the server.", param="response_id"
                    )
                parent = payload.get("previous_response_id")
                if parent is not None:
                    if not isinstance(parent, str) or not parent.strip():
                        raise RequestValidationError(
                            "previous_response_id must be a nonempty string.",
                            param="previous_response_id",
                        )
                    if payload.get("conversation") is not None:
                        raise RequestValidationError(
                            "previous_response_id and conversation "
                            "are mutually exclusive.",
                            param="previous_response_id",
                        )
            parsed = parse_and_validate_create_response(payload)
            if parsed.get("previous_response_id") is not None:
                context = ResponseContext(
                    response_id=parsed["previous_response_id"],
                    request=parsed,
                    mode_flags=ResponseModeFlags(
                        stream=False, store=True, background=False
                    ),
                    provider=self.provider,
                    platform_context=platform,
                )
                await select_branch_checkpoint(parsed, context, self.graph)
            return None
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            RequestValidationError,
        ) as exc:
            return error_response(exc, {})
        except FoundryResourceNotFoundError as exc:
            if exc.response_body is not None:
                return JSONResponse(
                    exc.response_body, status_code=404, headers={ERROR_SOURCE: "user"}
                )
            return not_found_response(str(parsed.get("previous_response_id")), {})
        except FoundryBadRequestError as exc:
            if exc.response_body is not None:
                return JSONResponse(
                    exc.response_body, status_code=400, headers={ERROR_SOURCE: "user"}
                )
            return invalid_request_response(str(exc), {})
        except FoundryApiError as exc:
            logger.exception("Responses branching admission failed")
            if exc.response_body is not None:
                return JSONResponse(
                    exc.response_body,
                    status_code=500,
                    headers={ERROR_SOURCE: "platform"},
                )
            return error_response(exc, {})
        except Exception:
            logger.exception("Responses branching admission failed")
            return error_response(RuntimeError("Unable to restore the parent."), {})
