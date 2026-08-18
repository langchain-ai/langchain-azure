# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""LangGraph runnable config management for Responses API hosting."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from azure.ai.agentserver.responses import ResponseContext
from langchain_core.runnables import RunnableConfig

from .checkpoint_ref import CheckpointRef


class HostingRunnableConfig:
    """Wrap hosting-owned data stored in a LangGraph runnable config."""

    def __init__(self, config: RunnableConfig) -> None:
        self._config = config

    @classmethod
    def create(
        cls,
        thread_id: str,
        response_context: ResponseContext,
    ) -> HostingRunnableConfig:
        """Create a config for one Responses handler invocation."""
        return cls(
            cast(
                RunnableConfig,
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "response_context": response_context,
                    }
                },
            )
        )

    @classmethod
    def create_from_checkpoint(
        cls,
        checkpoint_ref: CheckpointRef,
        response_context: ResponseContext,
    ) -> HostingRunnableConfig:
        """Create a config pinned to an exact LangGraph checkpoint."""
        return cls(
            cast(
                RunnableConfig,
                {
                    "configurable": {
                        "thread_id": checkpoint_ref.thread_id,
                        "checkpoint_id": checkpoint_ref.checkpoint_id,
                        "checkpoint_ns": "",
                        "response_context": response_context,
                    }
                },
            )
        )

    @property
    def runnable_config(self) -> RunnableConfig:
        """Return the wrapped LangGraph config."""
        return self._config

    @property
    def checkpoint_ref(self) -> CheckpointRef | None:
        """Return the LangGraph reference in the wrapped config."""
        configurable = self._config.get("configurable") or {}
        thread_id = configurable.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            return None
        checkpoint_id = configurable.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            return None
        return CheckpointRef(thread_id, checkpoint_id)

    def with_checkpoint_ref(
        self,
        checkpoint_ref: CheckpointRef,
    ) -> HostingRunnableConfig:
        """Return a copy pinned to the supplied LangGraph reference."""
        configurable = self._configurable_copy()
        configurable["thread_id"] = checkpoint_ref.thread_id
        configurable["checkpoint_id"] = checkpoint_ref.checkpoint_id
        configurable["checkpoint_ns"] = ""
        return self._with_configurable(configurable)

    def with_response_context(
        self,
        response_context: ResponseContext,
    ) -> HostingRunnableConfig:
        """Return a copy carrying the current Responses context."""
        configurable = self._configurable_copy()
        configurable["response_context"] = response_context
        return self._with_configurable(configurable)

    def with_cancellation_signal(
        self,
        cancellation_signal: asyncio.Event,
    ) -> HostingRunnableConfig:
        """Return a copy carrying the current response cancellation signal."""
        configurable = self._configurable_copy()
        configurable["response_cancellation_signal"] = cancellation_signal
        return self._with_configurable(configurable)

    def _configurable_copy(self) -> dict[str, Any]:
        return dict(self._config.get("configurable") or {})

    def _with_configurable(
        self,
        configurable: dict[str, Any],
    ) -> HostingRunnableConfig:
        return HostingRunnableConfig(
            cast(
                RunnableConfig,
                {
                    **self._config,
                    "configurable": configurable,
                },
            )
        )
