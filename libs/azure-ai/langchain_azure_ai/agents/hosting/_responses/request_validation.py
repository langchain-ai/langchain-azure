"""Validate Responses requests before HTTP streaming or graph execution."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any

from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    read_request_id,
    reset_request_context,
    set_request_context,
)
from azure.ai.agentserver.responses import ResponseContext, ResponseEventStream
from azure.ai.agentserver.responses.models import CreateResponse, ResponseObject
from azure.ai.agentserver.responses.store._foundry_errors import (
    FoundryResourceNotFoundError,
)
from azure.core.exceptions import ResourceNotFoundError
from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, request_response

from .._storage import state_store_client_scope
from .checkpoint_ref import CheckpointRef
from .hosting_runnable_config import HostingRunnableConfig
from .task_storage_manager import TaskStorageManager

if TYPE_CHECKING:
    from .._responses_host import ResponsesHostServer

logger = logging.getLogger(__name__)
_request_id: ContextVar[str | None] = ContextVar(
    "responses_validation_request_id", default=None
)


class _RequestValidationError(Exception):
    """An admission failure with an OpenAI-compatible error payload."""

    def __init__(
        self,
        message: str,
        param: str | None,
        *,
        status_code: int = 400,
        code: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = {
            "message": message,
            "type": "server_error" if status_code >= 500 else "invalid_request_error",
            "code": code,
            "param": param,
        }


def _error_response(
    error: _RequestValidationError, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    details: dict[str, Any] = dict(error.error)
    if request_id := _request_id.get():
        details["additionalInfo"] = {"request_id": request_id}
    return JSONResponse(
        {"error": details},
        status_code=error.status_code,
        headers={
            **(headers or {}),
            "x-platform-error-source": (
                "platform" if error.status_code >= 500 else "user"
            ),
        },
    )


async def _load_parent_response(
    parent_id: str, context: ResponseContext
) -> ResponseObject:
    provider = getattr(context, "_provider", None)
    if provider is None:
        raise RuntimeError("Response validation requires a response provider.")
    try:
        parent = await provider.get_response(
            parent_id, context=context.platform_context
        )
    except (KeyError, FoundryResourceNotFoundError, ResourceNotFoundError) as exc:
        raise _RequestValidationError(
            "The previous response was not found.",
            "previous_response_id",
            status_code=404,
            code="previous_response_not_found",
        ) from exc
    if parent is None or parent.get("id") != parent_id:
        raise _RequestValidationError(
            "The previous response was not found.",
            "previous_response_id",
            status_code=404,
            code="previous_response_not_found",
        )
    return parent


def _validate_branch_parent(
    parent_id: str,
    parent: ResponseObject,
    context: ResponseContext,
) -> CheckpointRef:
    """Validate a branch parent and return its checked checkpoint reference."""
    from .._responses_host import _scope_thread_id, _thread_namespace

    if parent.get("status") != "completed":
        raise _RequestValidationError(
            "The previous response must be completed.", "previous_response_id"
        )

    reference = TaskStorageManager.from_stream(
        ResponseEventStream(response_id=parent_id, response=parent)
    ).checkpoint_ref
    if reference is None:
        raise _RequestValidationError(
            "The previous response has no saved graph checkpoint.",
            "previous_response_id",
        )

    if _thread_namespace(reference.thread_id) != _scope_thread_id("", context):
        raise _RequestValidationError(
            "The previous response was not found.",
            "previous_response_id",
            status_code=404,
            code="previous_response_not_found",
        )
    return reference


def _validate_branch_state(state: StateSnapshot, config: RunnableConfig) -> None:
    """Require the selected checkpoint to exist and contain a finished graph."""
    if (
        HostingRunnableConfig(state.config).checkpoint_ref
        != HostingRunnableConfig(config).checkpoint_ref
        or (state.config.get("configurable") or {}).get("checkpoint_ns", "") != ""
        or state.created_at is None
    ):
        raise _RequestValidationError(
            "The previous response's graph checkpoint is missing.",
            "previous_response_id",
        )
    if state.next or state.tasks or state.interrupts:
        raise _RequestValidationError(
            "Branching requires a finished parent graph.", "previous_response_id"
        )


async def _validate_request(
    server: ResponsesHostServer, request: CreateResponse, context: ResponseContext
) -> None:
    parent_id = request.get("previous_response_id")
    if not isinstance(parent_id, str) or not parent_id:
        return
    parent = await _load_parent_response(parent_id, context)
    if not server._is_response_branch(request, context):
        return
    _validate_branch_parent(parent_id, parent, context)
    config = await server.build_runnable_config(request, context)
    _validate_branch_state(await server.graph.aget_state(config), config)


def install_request_validation(server: ResponsesHostServer) -> None:
    """Install early parameter checks and context-aware SDK preflight checks."""
    app = server.app
    endpoint = app._endpoint
    prefetch = endpoint._prefetch_history_ids

    @wraps(prefetch)
    async def preflight(
        ctx: Any, *, span: Any, agent_session_id: str | None
    ) -> Response | None:
        response = await prefetch(ctx, span=span, agent_session_id=agent_session_id)
        if response is not None:
            return response
        token = set_request_context(
            FoundryAgentRequestContext(
                user_id=ctx.user_id,
                call_id=ctx.call_id,
                session_id=ctx.agent_session_id,
            )
        )
        try:
            async with state_store_client_scope():
                await _validate_request(server, ctx.parsed, ctx.context)
        except _RequestValidationError as exc:
            span.end(exc)
            return _error_response(exc, endpoint._session_headers(agent_session_id))
        except Exception as exc:
            logger.exception("Responses request validation failed")
            span.end(exc)
            return _error_response(
                _RequestValidationError(
                    "An internal error occurred.",
                    None,
                    status_code=500,
                    code="server_error",
                ),
                endpoint._session_headers(agent_session_id),
            )
        finally:
            reset_request_context(token)
        return None

    setattr(endpoint, "_prefetch_history_ids", preflight)
    for route in app.routes:
        if not isinstance(route, Route) or route.name != "create_response":
            continue
        create_response = route.endpoint

        @wraps(create_response)
        async def validate_store(request: Request) -> Response:
            token = _request_id.set(read_request_id(request.scope))
            try:
                try:
                    payload = await request.json()
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return await create_response(request)
                if isinstance(payload, dict) and payload.get("store") is False:
                    return _error_response(
                        _RequestValidationError(
                            "store=False is not supported yet.", "store"
                        )
                    )
                return await create_response(request)
            finally:
                _request_id.reset(token)

        route.endpoint = validate_store
        route.app = request_response(validate_store)
        return
