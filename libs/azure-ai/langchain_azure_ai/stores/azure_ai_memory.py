"""Azure AI Memory Store for LangGraph."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

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

if TYPE_CHECKING:
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import MemoryStoreSearchResult

logger = logging.getLogger(__name__)

_NAMESPACE_SEP = "/"


class AzureAIMemoryStore(BaseStore):
    """LangGraph ``BaseStore`` backed by Azure AI Projects memory stores.

    Uses the Azure AI Projects SDK V2 ``beta.memory_stores`` API to persist
    and semantically search memories. Both synchronous ``batch()`` and
    asynchronous ``abatch()`` are supported.

    !!! example "Examples"
        Basic usage with an existing memory store:
        ```python
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
        from langchain_azure_ai.stores import AzureAIMemoryStore

        project_client = AIProjectClient(
            endpoint="https://<your-endpoint>",
            credential=DefaultAzureCredential(),
        )
        store = AzureAIMemoryStore(
            project_client=project_client,
            memory_store_name="my-memory-store",
        )

        # Store a memory
        store.put(("users", "alice"), "preferences", {"theme": "dark"})

        # Search for relevant memories
        results = store.search(("users", "alice"), query="user preferences")
        ```

    Note:
        The Azure AI memory store processes conversation items to extract
        semantic memories. Exact key-value retrieval (``get``) is provided on
        a best-effort basis via semantic search. The ``list_namespaces``
        operation is not supported and always returns an empty list.

    Note:
        ``azure-ai-projects >= 2.0.0b4`` is required. Install with:
        ``pip install 'azure-ai-projects>=2.0.0b4' --pre``
    """

    def __init__(
        self,
        project_client: AIProjectClient,
        memory_store_name: str,
    ) -> None:
        """Initialize an ``AzureAIMemoryStore``.

        Args:
            project_client: An authenticated ``AIProjectClient`` instance.
            memory_store_name: The name of the Azure AI memory store to use.
                The store must already exist in the Azure AI project.
        """
        try:
            from azure.ai.projects import (
                AIProjectClient as _AIProjectClient,  # noqa: F401
            )
        except ImportError as exc:
            raise ImportError(
                "azure-ai-projects>=2.0.0b4 is required. "
                "Install with: pip install 'azure-ai-projects>=2.0.0b4' --pre"
            ) from exc
        self._client = project_client
        self._memory_store_name = memory_store_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _namespace_to_scope(self, namespace: tuple[str, ...]) -> str:
        """Convert a namespace tuple to an Azure AI scope string."""
        return _NAMESPACE_SEP.join(namespace)

    def _scope_to_namespace(self, scope: str) -> tuple[str, ...]:
        """Convert an Azure AI scope string to a namespace tuple."""
        return tuple(scope.split(_NAMESPACE_SEP))

    def _make_message(self, key: str, value: dict[str, Any]) -> dict[str, str]:
        """Serialize a key-value pair as an Azure AI conversation message."""
        content = json.dumps({"key": key, "value": value})
        return {"role": "user", "type": "message", "content": content}

    def _parse_memory_content(
        self, content: str
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Try to parse a memory content string as a structured key-value pair.

        Returns:
            A ``(key, value)`` tuple if parsing succeeds, otherwise
            ``(None, None)``.
        """
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "key" in data and "value" in data:
                return str(data["key"]), data["value"]
        except (json.JSONDecodeError, ValueError):
            pass
        return None, None

    def _search_result_to_items(
        self,
        result: MemoryStoreSearchResult,
        namespace_prefix: tuple[str, ...],
        limit: int,
        offset: int,
        filter: Optional[dict[str, Any]],
    ) -> list[SearchItem]:
        """Convert an Azure AI search result to a list of ``SearchItem`` objects."""
        items: list[SearchItem] = []
        for search_item in result.memories:
            memory = search_item.memory_item
            namespace = self._scope_to_namespace(memory.scope)

            # Only return items whose namespace starts with the requested prefix.
            if namespace[: len(namespace_prefix)] != namespace_prefix:
                continue

            key, value = self._parse_memory_content(memory.content)
            if value is None:
                # The AI may have reformatted the content; fall back to raw text.
                value = {"content": memory.content}
                key = memory.memory_id

            if filter and not all(
                value.get(k) == v for k, v in filter.items()
            ):
                continue

            items.append(
                SearchItem(
                    namespace=namespace,
                    key=key,  # type: ignore[arg-type]
                    value=value,
                    created_at=memory.updated_at,
                    updated_at=memory.updated_at,
                )
            )

        return items[offset : offset + limit]

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def _do_put(self, op: PutOp) -> None:
        """Execute a single ``PutOp`` synchronously."""
        scope = self._namespace_to_scope(op.namespace)
        if op.value is None:
            # Value=None means delete; remove all memories for this scope.
            try:
                self._client.beta.memory_stores.delete_scope(
                    name=self._memory_store_name,
                    scope=scope,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("delete_scope failed (scope=%s): %s", scope, exc)
            return

        message = self._make_message(op.key, op.value)
        poller = self._client.beta.memory_stores.begin_update_memories(
            name=self._memory_store_name,
            scope=scope,
            items=[message],
            update_delay=0,
        )
        poller.result()

    def _do_get(self, op: GetOp) -> Optional[Item]:
        """Execute a single ``GetOp`` synchronously."""
        from azure.ai.projects.models import MemorySearchOptions

        scope = self._namespace_to_scope(op.namespace)
        query_message = {
            "role": "user",
            "type": "message",
            "content": f"key: {op.key}",
        }
        result = self._client.beta.memory_stores.search_memories(
            name=self._memory_store_name,
            scope=scope,
            items=[query_message],
            options=MemorySearchOptions(max_memories=20),
        )
        for search_item in result.memories:
            memory = search_item.memory_item
            key, value = self._parse_memory_content(memory.content)
            if key == op.key:
                return Item(
                    value=value,  # type: ignore[arg-type]
                    key=op.key,
                    namespace=op.namespace,
                    created_at=memory.updated_at,
                    updated_at=memory.updated_at,
                )
        return None

    def _do_search(self, op: SearchOp) -> list[SearchItem]:
        """Execute a single ``SearchOp`` synchronously."""
        from azure.ai.projects.models import MemorySearchOptions

        scope = self._namespace_to_scope(op.namespace_prefix)
        query = op.query or ""
        query_message = {
            "role": "user",
            "type": "message",
            "content": query,
        }
        result = self._client.beta.memory_stores.search_memories(
            name=self._memory_store_name,
            scope=scope,
            items=[query_message],
            options=MemorySearchOptions(max_memories=op.limit + op.offset),
        )
        return self._search_result_to_items(
            result,
            namespace_prefix=op.namespace_prefix,
            limit=op.limit,
            offset=op.offset,
            filter=op.filter,
        )

    # ------------------------------------------------------------------
    # BaseStore abstract methods
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of store operations synchronously.

        Args:
            ops: An iterable of ``GetOp``, ``PutOp``, ``SearchOp``, or
                ``ListNamespacesOp`` instances.

        Returns:
            A list of results, one per operation. ``PutOp`` results are
            ``None``; ``ListNamespacesOp`` results are empty lists.
        """
        results: list[Result] = []
        for op in ops:
            if isinstance(op, PutOp):
                self._do_put(op)
                results.append(None)
            elif isinstance(op, GetOp):
                results.append(self._do_get(op))
            elif isinstance(op, SearchOp):
                results.append(self._do_search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of store operations asynchronously.

        Args:
            ops: An iterable of ``GetOp``, ``PutOp``, ``SearchOp``, or
                ``ListNamespacesOp`` instances.

        Returns:
            A list of results, one per operation. ``PutOp`` results are
            ``None``; ``ListNamespacesOp`` results are empty lists.
        """
        loop = asyncio.get_event_loop()
        results: list[Result] = []
        for op in ops:
            if isinstance(op, PutOp):
                await loop.run_in_executor(None, self._do_put, op)
                results.append(None)
            elif isinstance(op, GetOp):
                item = await loop.run_in_executor(None, self._do_get, op)
                results.append(item)
            elif isinstance(op, SearchOp):
                items = await loop.run_in_executor(None, self._do_search, op)
                results.append(items)
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return results

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Handle a ``ListNamespacesOp``.

        The Azure AI memory store does not expose an API to enumerate scopes,
        so this always returns an empty list.
        """
        return []
