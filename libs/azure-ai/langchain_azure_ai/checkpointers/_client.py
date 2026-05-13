# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Async client for the Azure AI Foundry checkpoint storage REST API.

This client is vendored from ``azure-ai-agentserver-core`` so that
``langchain-azure-ai`` does not need a runtime dependency on the
agent server SDK. The REST contract (api-version ``2025-11-15-preview``)
is owned by the Foundry service.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, AsyncContextManager, List, Optional

from azure.core import AsyncPipelineClient
from azure.core.configuration import Configuration
from azure.core.credentials_async import AsyncTokenCredential
from azure.core.pipeline import policies
from azure.core.tracing.decorator_async import distributed_trace_async

from ._models import CheckpointItem, CheckpointItemId, CheckpointSession
from ._operations import CheckpointItemOperations, CheckpointSessionOperations

_PACKAGE_NAME = "langchain-azure-ai"
try:
    _PACKAGE_VERSION = importlib.metadata.version(_PACKAGE_NAME)
except importlib.metadata.PackageNotFoundError:
    _PACKAGE_VERSION = "0.0.0"
_USER_AGENT = f"{_PACKAGE_NAME}/{_PACKAGE_VERSION} foundry-checkpoint"


class _FoundryCheckpointClientConfiguration(Configuration):
    """Pipeline configuration for the Foundry checkpoint client."""

    def __init__(self, credential: "AsyncTokenCredential") -> None:
        super().__init__()
        self.retry_policy = policies.AsyncRetryPolicy()
        self.logging_policy = policies.NetworkTraceLoggingPolicy()
        self.request_id_policy = policies.RequestIdPolicy()
        self.http_logging_policy = policies.HttpLoggingPolicy()
        self.user_agent_policy = policies.UserAgentPolicy(base_user_agent=_USER_AGENT)
        self.authentication_policy = policies.AsyncBearerTokenCredentialPolicy(
            credential, "https://ai.azure.com/.default"
        )
        self.redirect_policy = policies.AsyncRedirectPolicy()


class FoundryCheckpointClient(AsyncContextManager["FoundryCheckpointClient"]):
    """Asynchronous client for the Azure AI Foundry checkpoint storage API.

    Provides session and item operations against the Foundry checkpoint
    service. This is an internal transport used by
    :class:`langchain_azure_ai.checkpointers.FoundryCheckpointSaver`.

    :param endpoint: The fully qualified project endpoint for the Azure AI
        Foundry service, for example
        ``"https://<resource>.services.ai.azure.com/api/projects/<project-id>"``.
    :type endpoint: str
    :param credential: Credential for authenticating requests. Use an async
        credential from ``azure.identity.aio``.
    :type credential: ~azure.core.credentials_async.AsyncTokenCredential
    """

    def __init__(
        self,
        endpoint: str,
        credential: "AsyncTokenCredential",
    ) -> None:
        config = _FoundryCheckpointClientConfiguration(credential)
        self._client: AsyncPipelineClient = AsyncPipelineClient(
            base_url=endpoint, config=config
        )
        self._sessions = CheckpointSessionOperations(self._client)
        self._items = CheckpointItemOperations(self._client)

    # Session operations

    @distributed_trace_async
    async def upsert_session(self, session: CheckpointSession) -> CheckpointSession:
        """Create or update a checkpoint session."""
        return await self._sessions.upsert(session)

    @distributed_trace_async
    async def read_session(self, session_id: str) -> Optional[CheckpointSession]:
        """Read a checkpoint session by ID. Returns ``None`` if not found."""
        return await self._sessions.read(session_id)

    @distributed_trace_async
    async def delete_session(self, session_id: str) -> None:
        """Delete a checkpoint session."""
        await self._sessions.delete(session_id)

    # Item operations

    @distributed_trace_async
    async def create_items(self, items: List[CheckpointItem]) -> List[CheckpointItem]:
        """Create checkpoint items in batch."""
        return await self._items.create_batch(items)

    @distributed_trace_async
    async def read_item(self, item_id: CheckpointItemId) -> Optional[CheckpointItem]:
        """Read a checkpoint item by ID. Returns ``None`` if not found."""
        return await self._items.read(item_id)

    @distributed_trace_async
    async def delete_item(self, item_id: CheckpointItemId) -> bool:
        """Delete a checkpoint item. Returns ``False`` if not found."""
        return await self._items.delete(item_id)

    @distributed_trace_async
    async def list_item_ids(
        self, session_id: str, parent_id: Optional[str] = None
    ) -> List[CheckpointItemId]:
        """List checkpoint item IDs for a session."""
        return await self._items.list_ids(session_id, parent_id)

    # Context manager / lifecycle

    async def close(self) -> None:
        """Close the underlying HTTP pipeline."""
        await self._client.close()

    async def __aenter__(self) -> "FoundryCheckpointClient":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc_details: Any) -> None:
        await self._client.__aexit__(*exc_details)
