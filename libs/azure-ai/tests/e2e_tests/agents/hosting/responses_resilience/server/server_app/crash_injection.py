# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Test-only crash injection at Responses and persisted checkpoint boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from azure.ai.agentserver.responses import CreateResponse, ResponseContext
from azure.ai.agentserver.responses.streaming._checkpoint import ResponseCheckpointEvent

from langchain_azure_ai.agents.hosting import ResponsesHostServer

from .crash_helpers import (
    crash_marker_exists,
    crash_once,
    event_checkpoint_ref,
)
from .crash_points import (
    AFTER_1PLAN_RESPONSES_CHECKPOINT,
    AFTER_2RESEARCH_CHECKPOINT_BEFORE_METADATA,
    AFTER_2RESEARCH_GRAPH_CHECKPOINT_BEFORE_METADATA,
    AFTER_FINAL_RESPONSES_CHECKPOINT_BEFORE_STATE_STORE,
    AFTER_RESPONSE_COMPLETED,
    AFTER_RESPONSE_CREATED_BEFORE_FIRST_CHECKPOINT,
    AFTER_STATE_STORE_BEFORE_COMPLETED,
    BEFORE_RESPONSE_CREATED,
    NEW_CONVERSATION_FIRST_CHECKPOINT_BEFORE_METADATA,
    SECOND_TURN_FIRST_CHECKPOINT_BEFORE_METADATA,
    request_crash_point,
)
from .workflow import RESEARCH_OUTPUT

_TURN_ONE_STATE_WRITES = {
    "plan_writes": 1,
    "research_writes": 1,
    "execute_writes": 1,
    "summarize_writes": 1,
}


class CrashInjectingResponsesHostServer(ResponsesHostServer):
    """Install test-only crash hooks around graph and Responses boundaries."""

    async def _checkpoint_state(
        self,
        event: ResponseCheckpointEvent,
    ) -> tuple[tuple[str, str] | None, dict[str, Any]]:
        checkpoint = event_checkpoint_ref(event)
        if checkpoint is None:
            return None, {}
        thread_id, checkpoint_id = checkpoint
        snapshot = await self.graph.aget_state(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }
        )
        values = snapshot.values
        return checkpoint, values if isinstance(values, dict) else {}

    async def handle_create(
        self,
        request: CreateResponse,
        context: ResponseContext,
        cancellation_signal: asyncio.Event,
    ) -> AsyncIterator[Any]:
        crash_point = request_crash_point(request)
        if crash_point == BEFORE_RESPONSE_CREATED:
            crash_once(
                (context.conversation_id or context.response_id, None),
                crash_point,
            )
        async for event in super().handle_create(
            request,
            context,
            cancellation_signal,
        ):
            checkpoint_ref: tuple[str, str] | None = None
            checkpoint_values: dict[str, Any] = {}
            if isinstance(event, ResponseCheckpointEvent):
                checkpoint_ref, checkpoint_values = await self._checkpoint_state(event)
            is_first_research_token = (
                isinstance(event, dict)
                and event.get("type") == "response.output_text.delta"
                and event.get("delta") == f"{RESEARCH_OUTPUT.split(' ')[0]} "
            )
            is_completed = (
                isinstance(event, dict) and event.get("type") == "response.completed"
            )
            is_admission_checkpoint = (
                isinstance(event, ResponseCheckpointEvent) and checkpoint_ref is None
            )
            stage = checkpoint_values.get("stage")
            checkpoint_writes = {
                field: checkpoint_values.get(field, 0)
                for field in _TURN_ONE_STATE_WRITES
            }
            if (
                crash_point == NEW_CONVERSATION_FIRST_CHECKPOINT_BEFORE_METADATA
                and checkpoint_ref is not None
                and checkpoint_writes == {field: 0 for field in _TURN_ONE_STATE_WRITES}
            ):
                crash_once(checkpoint_ref, crash_point)
            if (
                crash_point == SECOND_TURN_FIRST_CHECKPOINT_BEFORE_METADATA
                and checkpoint_ref is not None
                and checkpoint_writes == _TURN_ONE_STATE_WRITES
            ):
                if not crash_marker_exists(checkpoint_ref[0]):
                    assert checkpoint_values.get("plan_writes", 0) == 1
                crash_once(checkpoint_ref, crash_point)
            if (
                crash_point == AFTER_2RESEARCH_GRAPH_CHECKPOINT_BEFORE_METADATA
                and stage == "researched"
                and checkpoint_ref is not None
            ):
                crash_once(checkpoint_ref, crash_point)
            if crash_point == AFTER_STATE_STORE_BEFORE_COMPLETED and is_completed:
                crash_once((context.response_id, None), crash_point)
            yield event
            if (
                crash_point == AFTER_RESPONSE_CREATED_BEFORE_FIRST_CHECKPOINT
                and is_admission_checkpoint
            ):
                crash_once((context.response_id, None), crash_point)
            if crash_point == AFTER_RESPONSE_COMPLETED and is_completed:
                crash_once((context.response_id, None), crash_point)
            if (
                crash_point == AFTER_2RESEARCH_CHECKPOINT_BEFORE_METADATA
                and is_first_research_token
            ):
                crash_once((context.response_id, None), crash_point)
            # Resumption after yield means the orchestrator's awaited
            # _do_checkpoint_persist call completed for this exact snapshot.
            if (
                crash_point == AFTER_1PLAN_RESPONSES_CHECKPOINT
                and stage == "planned"
                and checkpoint_ref is not None
            ):
                crash_once(checkpoint_ref, crash_point)
            if (
                crash_point == AFTER_FINAL_RESPONSES_CHECKPOINT_BEFORE_STATE_STORE
                and checkpoint_values.get("summarize_writes", 0) > 0
                and checkpoint_ref is not None
            ):
                crash_once(checkpoint_ref, crash_point)
