"""AzureAIMemoryStore: LangGraph BaseStore backed by Azure AI Memory Stores (V2).

This module provides ``AzureAIMemoryStore``, a persisted LangGraph store that
uses the Azure AI Projects SDK V2 (``azure-ai-projects>=2.0.0b4``) to store and
retrieve memories via Azure AI Foundry.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from azure.core.exceptions import HttpResponseError
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from langchain_azure_ai._api.base import experimental

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

logger = logging.getLogger(__name__)

NAMESPACE_AUTHENTICATED_USER = ("{{$userId}}",)
"""Namespace for storing and partitioning authenticated user profiles in Azure AI 
memory stores."""

CONTENT_KEY = "content"
"""Key name for the main content of a memory item.  Azure AI Memory Stores only 
support values with a "content" key, which is stored directly as the memory text and
is what the service will search and return."""


@experimental()
class AzureAIMemoryStore(BaseStore):
    """A persisted LangGraph ``BaseStore`` backed by Azure AI Memory Stores.

    This store uses the Azure AI Projects SDK V2 to persist memories in Azure AI
    Foundry. It maps LangGraph's namespace/key semantics onto Azure AI
    memory scopes and stores values as structured natural-language messages
    that the underlying AI can extract and search.

    !!! warning "Preview feature"
        Azure AI Memory Stores are currently in preview. This class is
        experimental and may change without notice.

    !!! note "Limitations"
        - ``list_namespaces`` always returns an empty list because Azure AI
          memory stores do not expose a namespace enumeration API.
        - Deleting a key (``put`` with ``value=None``) removes **all**
          memories in the entire namespace scope, not just the one key.
        - Only values with a ``"content"`` key are supported.  The value of
          ``content`` is stored directly as the memory text, which is what
          the service will search and return.

    Note:
        Namespaces must contain exactly **one** element.  Azure AI memory
        stores use a flat scope model and do not support hierarchical
        namespaces.

    Args:
        memory_store_name: Name of an existing Azure AI memory store.
        project_endpoint: Azure AI project endpoint URL.  Falls back to the
            ``AZURE_AI_PROJECT_ENDPOINT`` environment variable.
        credential: Azure credential.  Defaults to
            ``DefaultAzureCredential`` when not provided.
        api_version: Optional API version override for the
            ``AIProjectClient``.
        client_kwargs: Additional keyword arguments forwarded to
            ``AIProjectClient()``.

    Example:
        ```python
        from langchain_azure_ai.stores.memory import AzureAIMemoryStore
        from azure.identity import DefaultAzureCredential

        store = AzureAIMemoryStore(
            memory_store_name="my-memory-store",
            project_endpoint="https://my-resource.services.ai.azure.com/api/projects/my-project",
            credential=DefaultAzureCredential(),
        )

        # Store a value – the value dict must contain a "content" key.
        store.put(("user_alice",), "content", {"content": "prefers dark theme"})

        # Retrieve it
        item = store.get(("user_alice",), "content")
        if item:
            print(item.value)  # {"content": "prefers dark theme"}

        # Semantic search
        results = store.search(("user_alice",), query="user preferences")
        ```
    """

    def __init__(
        self,
        memory_store_name: str,
        *,
        project_endpoint: Optional[str] = None,
        credential: Optional["TokenCredential"] = None,
        api_version: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the store.

        Args:
            memory_store_name: Name of an existing Azure AI memory store.
            project_endpoint: Azure AI project endpoint.  Falls back to
                ``AZURE_AI_PROJECT_ENDPOINT`` env var.
            credential: Azure credential to use.  Defaults to
                ``DefaultAzureCredential``.
            api_version: Optional API version override for the
                ``AIProjectClient``.
            client_kwargs: Additional keyword arguments forwarded to
                ``AIProjectClient()``.
        """
        import os

        from azure.ai.projects import AIProjectClient as _AIProjectClient
        from azure.identity import DefaultAzureCredential

        resolved_endpoint = project_endpoint or os.environ.get(
            "AZURE_AI_PROJECT_ENDPOINT"
        )
        if not resolved_endpoint:
            raise ValueError(
                "A 'project_endpoint' must be provided, or the "
                "'AZURE_AI_PROJECT_ENDPOINT' environment variable must be set."
            )
        resolved_credential = (
            credential if credential is not None else DefaultAzureCredential()
        )
        init_kwargs: Dict[str, Any] = dict(client_kwargs or {})
        init_kwargs.setdefault("user_agent", "langchain-azure-ai")
        if api_version:
            init_kwargs["api_version"] = api_version
        resolved_client = _AIProjectClient(
            endpoint=resolved_endpoint,
            credential=resolved_credential,
            **init_kwargs,
        )

        if not hasattr(resolved_client, "beta") or not hasattr(
            resolved_client.beta, "memory_stores"
        ):
            raise ValueError(
                "The provided AIProjectClient does not support the memory "
                "stores API. azure-ai-projects>=2.0.0b4 is required. "
                "Install with: pip install 'azure-ai-projects>=2.0.0b4' --pre"
            )

        self._memory_store_name = memory_store_name
        self._client = resolved_client

    def create_memory_store(
        self,
        chat_model: str,
        embedding_model: str,
        description: Optional[str] = None,
        user_profile_instructions: Optional[str] = None,
    ) -> None:
        """Create the memory store in Azure AI Foundry if it doesn't already exist.

        Args:
            chat_model: The name of the chat model to use for this memory store.
            embedding_model: The name of the embedding model to use for this memory
                store.
            description: Optional description for the memory store.
            user_profile_instructions: Optional instructions for the user profile 
                extractor. If not provided, userprofile extraction will be enabled
                with default settings.

        Example:
            ```python
            store.create_memory_store(
                chat_model="gpt-4.1",
                embedding_model="text-embedding-3-small",
                description="My memory store",
                user_profile_instructions="Extract the user's name, location, and \
                    preferences from the conversation.",
            )
            ```
        """
        from azure.ai.projects.models import (
            MemoryStoreDefaultDefinition,
            MemoryStoreDefaultOptions,
        )

        definition = MemoryStoreDefaultDefinition(
            chat_model=chat_model,
            embedding_model=embedding_model,
            options=MemoryStoreDefaultOptions(
                user_profile_enabled=True,
                chat_summary_enabled=True,
                user_profile_details=user_profile_instructions,
            ),
        )
        _ = self._client.beta.memory_stores.create(
            name=self._memory_store_name,
            description=description,
            definition=definition,
        )
        logger.info(
            "Memory store '%s' created with chat model '%s' and embedding model '%s'.",
            self._memory_store_name,
            chat_model,
            embedding_model,
        )

    def delete_memory_store(self) -> None:
        """Delete the entire memory store from Azure AI Foundry.

        !!! warning
            This operation is irreversible and will delete all memories in the store.
        """
        try:
            self._client.beta.memory_stores.delete(name=self._memory_store_name)
            logger.info(
                "Memory store '%s' deleted successfully.", self._memory_store_name
            )
        except HttpResponseError as exc:
            logger.error(
                "Error deleting memory store '%s': %s", self._memory_store_name, exc
            )
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _namespace_to_scope(namespace: Tuple[str, ...]) -> str:
        """Convert a namespace tuple to an Azure AI memory store scope string.

        Raises:
            ValueError: If *namespace* does not contain exactly one element.
                Azure AI memory stores use a flat scope model and do not
                support hierarchical namespaces.
        """
        if len(namespace) != 1:
            raise ValueError(
                "AzureAIMemoryStore only supports namespaces with exactly one "
                f"element; got {len(namespace)}: {namespace!r}."
            )
        return namespace[0]

    @staticmethod
    def _make_message(content: str) -> Dict[str, Any]:
        """Build a conversation message from a *content* string."""
        return {"role": "user", "content": content, "type": "message"}

    # ------------------------------------------------------------------
    # Per-operation helpers
    # ------------------------------------------------------------------

    def _do_get(self, op: GetOp) -> Optional[Item]:
        """Handle a single GetOp."""
        from azure.ai.projects.models import MemorySearchOptions

        scope = self._namespace_to_scope(op.namespace)
        try:
            result = self._client.beta.memory_stores.search_memories(
                name=self._memory_store_name,
                scope=scope,
                items=[{"role": "user", "content": op.key, "type": "message"}],
                options=MemorySearchOptions(max_memories=1),
            )
        except HttpResponseError as exc:
            logger.debug(
                "Error searching memories for key '%s' in scope '%s': %s",
                op.key,
                scope,
                exc,
            )
            return None

        for mem_item in result.memories:
            memory = mem_item.memory_item
            timestamp = getattr(memory, "updated_at", None) or datetime.now(timezone.utc)
            return Item(
                namespace=op.namespace,
                key="content",
                value={"content": memory.content},
                created_at=timestamp,
                updated_at=timestamp,
            )
        return None

    def _do_put(self, op: PutOp) -> None:
        """Handle a single PutOp."""
        scope = self._namespace_to_scope(op.namespace)

        if op.value is None:
            # Delete all memories in this scope.
            try:
                self._client.beta.memory_stores.delete_scope(
                    name=self._memory_store_name,
                    scope=scope,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error deleting scope '%s': %s", scope, exc)
            return

        if not isinstance(op.value, Mapping):
            raise TypeError(
                "AzureAIMemoryStore only supports dict-like values with a 'content' key. "
                f"Got type: {type(op.value).__name__!r}"
            )
        if "content" not in op.value:
            raise ValueError(
                "AzureAIMemoryStore only supports values with a 'content' key. "
                f"Got keys: {list(op.value.keys())!r}"
            )

        message = self._make_message(op.value["content"])
        try:
            poller = self._client.beta.memory_stores.begin_update_memories(
                name=self._memory_store_name,
                scope=scope,
                items=[message],
                update_delay=0,
            )
            poller.result()
        except HttpResponseError as exc:
            logger.error(
                "Error updating memory for key '%s' in scope '%s': %s",
                op.key,
                scope,
                exc,
            )
            raise

    def _do_search(self, op: SearchOp) -> List[SearchItem]:
        """Handle a single SearchOp."""
        from azure.ai.projects.models import MemorySearchOptions

        scope = self._namespace_to_scope(op.namespace_prefix)
        query = op.query or ""
        # Azure AI memory stores do not support pagination natively, so we
        # fetch limit+offset memories and slice in Python.
        try:
            result = self._client.beta.memory_stores.search_memories(
                name=self._memory_store_name,
                scope=scope,
                items=[{"role": "user", "content": query, "type": "message"}],
                options=MemorySearchOptions(max_memories=op.limit + op.offset),
            )
        except HttpResponseError as exc:
            logger.debug("Error searching memories in scope '%s': %s", scope, exc)
            return []

        items: List[SearchItem] = []
        for mem_item in result.memories:
            memory = mem_item.memory_item
            timestamp = getattr(memory, "updated_at", None) or datetime.now(timezone.utc)
            items.append(
                SearchItem(
                    namespace=op.namespace_prefix,
                    key="content",
                    value={"content": memory.content},
                    created_at=timestamp,
                    updated_at=timestamp,
                    score=None,
                )
            )

        return items[op.offset : op.offset + op.limit]

    # ------------------------------------------------------------------
    # BaseStore interface
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> List[Result]:
        """Execute a batch of store operations synchronously.

        Args:
            ops: Iterable of ``GetOp``, ``PutOp``, ``SearchOp``, or
                ``ListNamespacesOp`` instances.

        Returns:
            A list of results matching the order of *ops*.
        """
        results: List[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._do_get(op))
            elif isinstance(op, PutOp):
                self._do_put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._do_search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        """Execute a batch of store operations asynchronously.

        Each operation is dispatched to a thread-pool executor individually
        so that I/O does not block the event loop.

        Args:
            ops: Iterable of store operations.

        Returns:
            A list of results matching the order of *ops*.
        """
        loop = asyncio.get_running_loop()
        results: List[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                item = await loop.run_in_executor(None, self._do_get, op)
                results.append(item)
            elif isinstance(op, PutOp):
                await loop.run_in_executor(None, self._do_put, op)
                results.append(None)
            elif isinstance(op, SearchOp):
                items = await loop.run_in_executor(None, self._do_search, op)
                results.append(items)
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return results

    def _list_namespaces(self, op: ListNamespacesOp) -> List[Tuple[str, ...]]:
        """Handle a ``ListNamespacesOp``.

        The Azure AI memory stores API does not provide a way to enumerate
        scopes, so this operation always returns an empty list.
        """
        return []
