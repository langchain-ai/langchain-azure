"""Exact checkpoint selection for completed, non-paused response branches."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from azure.ai.agentserver.responses import ResponseContext, ResponseProviderProtocol
from langchain_core.runnables import RunnableConfig
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .checkpoint_ref import CheckpointRef
from .task_storage_manager import CheckpointError as BranchingError
from .task_storage_manager import TaskStorageManager

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

BRANCH_MODE_HEADER = "x-client-langchain-response-branching"
BRANCH_MODE = "checkpoint-v1"
BRANCH_MODE_METADATA = "langgraph_response_branching"


_branch_selections: ContextVar[
    dict[tuple[int, ResponseContext, str], CheckpointRef] | None
] = ContextVar("langchain_azure_ai_branch_selections", default=None)


class BranchingAdmissionMiddleware:
    """Stamp trusted execution mode and validate opt-in response linkage.

    Args:
        app: The next ASGI application.
        enabled: Whether fresh requests may use response branching. Client mode
            headers are removed even when branching is disabled.
    """

    def __init__(self, app: ASGIApp, *, enabled: bool) -> None:
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Validate fresh HTTP requests before SDK admission."""
        if scope["type"] == "http":
            headers = [
                (name, value)
                for name, value in scope.get("headers", [])
                if name.lower() != BRANCH_MODE_HEADER.encode("ascii")
            ]
            if self.enabled:
                headers.append(
                    (BRANCH_MODE_HEADER.encode("ascii"), BRANCH_MODE.encode("ascii"))
                )
            scope = {**scope, "headers": headers}
            if (
                self.enabled
                and scope["method"] == "POST"
                and scope["path"].rstrip("/").endswith("/responses")
            ):
                body = await Request(scope, receive).body()
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
                original_receive = receive
                body_sent = False

                async def replay_body() -> Any:
                    nonlocal body_sent
                    if not body_sent:
                        body_sent = True
                        return {
                            "type": "http.request",
                            "body": body,
                            "more_body": False,
                        }
                    return await original_receive()

                receive = replay_body
        await self.app(scope, receive, send)

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


@contextmanager
def branch_selection_scope() -> Iterator[None]:
    """Reuse checkpoint selections only during one request's config preparation."""
    token = _branch_selections.set({})
    try:
        yield
    finally:
        _branch_selections.reset(token)


def _require_completed_tools(parent: Mapping[str, Any]) -> None:
    output = parent.get("output") or ()
    calls = {
        item.get("call_id") for item in output if item.get("type") == "function_call"
    }
    results = {
        item.get("call_id")
        for item in output
        if item.get("type") == "function_call_output"
    }
    if calls - results or any(
        item.get("type") == "mcp_approval_request" for item in output
    ):
        raise BranchingError(
            "unsupported_branch_parent",
            "Response branching does not support parents with pending "
            "approvals or tool results.",
        )


async def select_branch_checkpoint(
    provider: ResponseProviderProtocol | None,
    *,
    parent_id: str,
    context: ResponseContext,
) -> CheckpointRef:
    """Select the stored completed parent's exact checkpoint for fresh input."""
    if provider is None:
        provider = getattr(context, "_provider", None)
    selections = _branch_selections.get()
    key = (id(provider), context, parent_id)
    if selections is not None and key in selections:
        return selections[key]

    if provider is None:
        raise BranchingError(
            "invalid_branch_state", "The Responses provider is unavailable."
        )
    parent = await provider.get_response(parent_id, context=context.platform_context)
    if parent is None or parent.get("status") != "completed":
        raise BranchingError(
            "invalid_branch_state",
            "The parent response must be stored and completed.",
        )
    ref = TaskStorageManager.from_response(parent).checkpoint_ref
    if ref is None:
        raise BranchingError(
            "checkpoint_unavailable", "The parent has no completed checkpoint."
        )
    _require_completed_tools(parent)

    if selections is not None:
        selections[key] = ref
    return ref


async def validate_branch_checkpoint(
    graph: CompiledStateGraph, config: RunnableConfig
) -> None:
    """Validate the exact parent state after application config hooks return."""
    snapshot = await graph.aget_state(config)
    if snapshot.next or any(task.interrupts for task in snapshot.tasks):
        raise BranchingError(
            "unsupported_branch_parent",
            "Response branching does not support paused parent graphs.",
        )
