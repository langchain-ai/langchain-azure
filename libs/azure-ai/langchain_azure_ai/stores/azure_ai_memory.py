"""Azure AI Memory Store backed by Azure AI Projects SDK V2.

This module provides an implementation of LangGraph's ``BaseStore``
that persists data using the Azure AI Projects memory stores API.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import timezone
from typing import Any, Optional

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

logger = logging.getLogger(__name__)

# Scope delimiter used when joining namespace tuple elements into a scope string
_DEFAULT_SCOPE_DELIMITER = "/"


def _namespace_to_scope(
    namespace: tuple[str, ...],
    delimiter: str = _DEFAULT_SCOPE_DELIMITER,
) -> str:
    """Convert a namespace tuple to an Azure AI memory scope string.

    Args:
        namespace: Hierarchical namespace tuple, e.g. ``("user", "123")``.
        delimiter: Separator used when joining elements.

    Returns:
        A scope string, e.g. ``"user/123"``.
    """
    return delimiter.join(namespace)


def _scope_to_namespace(
    scope: str,
    delimiter: str = _DEFAULT_SCOPE_DELIMITER,
) -> tuple[str, ...]:
    """Convert an Azure AI memory scope string back to a namespace tuple.

    Args:
        scope: Scope string, e.g. ``"user/123"``.
        delimiter: Separator used when splitting.

    Returns:
        A namespace tuple, e.g. ``("user", "123")``.
    """
    return tuple(scope.split(delimiter))


def _memory_item_to_store_item(
    memory_item: Any,
    namespace: tuple[str, ...],
) -> Item:
    """Convert an Azure AI ``MemoryItem`` to a LangGraph ``Item``.

    Args:
        memory_item: An Azure AI ``MemoryItem`` object.
        namespace: Namespace tuple derived from the memory's scope.

    Returns:
        A LangGraph :class:`Item`.
    """
    updated = memory_item.updated_at
    if updated and not updated.tzinfo:
        updated = updated.replace(tzinfo=timezone.utc)
    return Item(
        value={
            "content": memory_item.content,
            "kind": str(memory_item.kind) if memory_item.kind else None,
        },
        key=memory_item.memory_id,
        namespace=namespace,
        created_at=updated,
        updated_at=updated,
    )


def _memory_item_to_search_item(
    memory_item: Any,
    namespace: tuple[str, ...],
    score: float | None = None,
) -> SearchItem:
    """Convert an Azure AI ``MemoryItem`` to a LangGraph ``SearchItem``.

    Args:
        memory_item: An Azure AI ``MemoryItem`` object.
        namespace: Namespace tuple derived from the memory's scope.
        score: Optional relevance score.

    Returns:
        A LangGraph :class:`SearchItem`.
    """
    updated = memory_item.updated_at
    if updated and not updated.tzinfo:
        updated = updated.replace(tzinfo=timezone.utc)
    return SearchItem(
        namespace=namespace,
        key=memory_item.memory_id,
        value={
            "content": memory_item.content,
            "kind": str(memory_item.kind) if memory_item.kind else None,
        },
        created_at=updated,
        updated_at=updated,
        score=score,
    )


def _build_message_items(value: dict[str, Any]) -> list[Any]:
    """Build Azure AI ``ItemParam`` objects from a value dictionary.

    The value dictionary may contain:
      - ``"content"`` – a text string used to create a user message.
      - ``"items"`` – a list of pre-built ``ItemParam`` objects.

    Args:
        value: A dictionary with the data to convert.

    Returns:
        A list of Azure AI ``ItemParam`` objects suitable for
        ``begin_update_memories`` or ``search_memories``.
    """
    from azure.ai.projects.models import (
        ResponsesUserMessageItemParam,
    )

    items = value.get("items")
    if items is not None:
        return list(items)

    content = value.get("content", "")
    if isinstance(content, str) and content:
        return [ResponsesUserMessageItemParam(content=content)]

    return []


class AzureAIMemoryStore(BaseStore):
    """LangGraph ``BaseStore`` backed by Azure AI Projects memory stores.

    This class bridges LangGraph's generic key-value store interface with the
    Azure AI Projects SDK V2 memory stores API, enabling persistent,
    AI-powered memory management across agent conversations.

    **Namespace → Scope mapping**:

    The namespace tuple is joined with a delimiter (default ``"/"``) to form an
    Azure AI memory *scope*.  For example ``("user", "123")`` becomes the scope
    ``"user/123"``.

    **Operation mapping**:

    +--------------------+------------------------------------------------------+
    | LangGraph Op       | Azure AI API call                                    |
    +====================+======================================================+
    | ``PutOp`` (value)  | ``begin_update_memories``                            |
    +--------------------+------------------------------------------------------+
    | ``PutOp`` (None)   | ``delete_scope``                                     |
    +--------------------+------------------------------------------------------+
    | ``SearchOp``       | ``search_memories``                                  |
    +--------------------+------------------------------------------------------+
    | ``GetOp``          | ``search_memories`` + client-side filter by key      |
    +--------------------+------------------------------------------------------+
    | ``ListNamespacesOp``| Returns scopes observed during the session           |
    +--------------------+------------------------------------------------------+

    Args:
        project_client: A *synchronous* ``AIProjectClient`` instance from the
            ``azure-ai-projects`` SDK (``>=2.0.0b1``).
        async_project_client: An optional *asynchronous* ``AIProjectClient``
            from ``azure.ai.projects.aio``.  Required only if you use
            ``abatch`` / ``aget`` / ``asearch`` etc.
        memory_store_name: The name of the Azure AI memory store to use.
        scope_delimiter: Separator for joining namespace elements into a scope
            string.  Defaults to ``"/"``.

    Example:
        .. code-block:: python

            from azure.ai.projects import AIProjectClient
            from azure.identity import DefaultAzureCredential

            project_client = AIProjectClient(
                credential=DefaultAzureCredential(),
                endpoint="https://your-endpoint.services.ai.azure.com/...",
            )

            store = AzureAIMemoryStore(
                project_client=project_client,
                memory_store_name="my-memory-store",
            )

            # Update memories from conversation content
            store.put(
                ("user123",), "conv1",
                {"content": "I love hiking and Python programming"},
            )

            # Search for relevant memories
            results = store.search(
                ("user123",),
                query="What are the user's hobbies?",
            )
    """

    def __init__(
        self,
        *,
        project_client: Any,
        async_project_client: Any | None = None,
        memory_store_name: str,
        scope_delimiter: str = _DEFAULT_SCOPE_DELIMITER,
    ) -> None:
        """Initialise the store.

        Args:
            project_client: Synchronous ``AIProjectClient`` instance.
            async_project_client: Optional async ``AIProjectClient``.
            memory_store_name: Name of the Azure AI memory store.
            scope_delimiter: Separator for namespace → scope conversion.
        """
        self._client = project_client
        self._async_client = async_project_client
        self._memory_store_name = memory_store_name
        self._scope_delimiter = scope_delimiter
        # Track observed scopes for list_namespaces support
        self._known_scopes: set[str] = set()

    # ------------------------------------------------------------------
    # BaseStore abstract interface
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations synchronously.

        Args:
            ops: An iterable of ``GetOp``, ``PutOp``, ``SearchOp``,
                or ``ListNamespacesOp`` operations.

        Returns:
            A list of results corresponding to each operation.
        """
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._handle_get(op))
            elif isinstance(op, SearchOp):
                results.append(self._handle_search(op))
            elif isinstance(op, PutOp):
                results.append(self._handle_put(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations asynchronously.

        Args:
            ops: An iterable of ``GetOp``, ``PutOp``, ``SearchOp``,
                or ``ListNamespacesOp`` operations.

        Returns:
            A list of results corresponding to each operation.
        """
        if self._async_client is None:
            raise ValueError(
                "An async_project_client is required for async operations. "
                "Provide one when creating AzureAIMemoryStore."
            )
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(await self._ahandle_get(op))
            elif isinstance(op, SearchOp):
                results.append(await self._ahandle_search(op))
            elif isinstance(op, PutOp):
                results.append(await self._ahandle_put(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return results

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def _handle_get(self, op: GetOp) -> Optional[Item]:
        """Handle a ``GetOp`` by searching for a memory with the given key.

        Args:
            op: The get operation.

        Returns:
            The matching :class:`Item` or ``None``.
        """
        from azure.ai.projects.models import (
            MemorySearchOptions,
            ResponsesUserMessageItemParam,
        )

        scope = _namespace_to_scope(op.namespace, self._scope_delimiter)
        self._known_scopes.add(scope)
        try:
            result = self._client.memory_stores.search_memories(
                name=self._memory_store_name,
                scope=scope,
                items=[ResponsesUserMessageItemParam(content="Retrieve all memories")],
                options=MemorySearchOptions(max_memories=100),
            )
        except Exception:
            logger.debug("Failed to search memories for get op", exc_info=True)
            return None

        for search_item in result.memories:
            mem = search_item.memory_item
            if mem.memory_id == op.key:
                namespace = _scope_to_namespace(
                    mem.scope, self._scope_delimiter
                )
                return _memory_item_to_store_item(mem, namespace)
        return None

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        """Handle a ``SearchOp`` using Azure AI ``search_memories``.

        Args:
            op: The search operation.

        Returns:
            A list of matching :class:`SearchItem` objects.
        """
        from azure.ai.projects.models import (
            MemorySearchOptions,
            ResponsesUserMessageItemParam,
        )

        scope = _namespace_to_scope(op.namespace_prefix, self._scope_delimiter)
        self._known_scopes.add(scope)

        query_text = op.query or "Retrieve all memories"
        items = [ResponsesUserMessageItemParam(content=query_text)]

        try:
            result = self._client.memory_stores.search_memories(
                name=self._memory_store_name,
                scope=scope,
                items=items,
                options=MemorySearchOptions(max_memories=op.limit + op.offset),
            )
        except Exception:
            logger.debug("Failed to search memories", exc_info=True)
            return []

        search_items: list[SearchItem] = []
        for search_result in result.memories:
            mem = search_result.memory_item
            namespace = _scope_to_namespace(mem.scope, self._scope_delimiter)

            if op.filter:
                value = {
                    "content": mem.content,
                    "kind": str(mem.kind) if mem.kind else None,
                }
                if not all(value.get(k) == v for k, v in op.filter.items()):
                    continue

            search_items.append(
                _memory_item_to_search_item(mem, namespace)
            )

        return search_items[op.offset : op.offset + op.limit]

    def _handle_put(self, op: PutOp) -> None:
        """Handle a ``PutOp``.

        When *value* is ``None`` the scope is deleted. Otherwise the value is
        fed to ``begin_update_memories`` so the service can extract and persist
        memories.

        Args:
            op: The put operation.
        """
        scope = _namespace_to_scope(op.namespace, self._scope_delimiter)
        self._known_scopes.add(scope)

        if op.value is None:
            try:
                self._client.memory_stores.delete_scope(
                    name=self._memory_store_name,
                    scope=scope,
                )
            except Exception:
                logger.debug("Failed to delete scope %s", scope, exc_info=True)
            self._known_scopes.discard(scope)
            return None

        items = _build_message_items(op.value)
        if not items:
            return None

        try:
            poller = self._client.memory_stores.begin_update_memories(
                name=self._memory_store_name,
                scope=scope,
                items=items,
                update_delay=0,
            )
            poller.result()
        except Exception:
            logger.debug(
                "Failed to update memories for scope %s", scope, exc_info=True
            )
        return None

    def _handle_list_namespaces(
        self,
        op: ListNamespacesOp,
    ) -> list[tuple[str, ...]]:
        """Handle a ``ListNamespacesOp``.

        Returns namespaces derived from scopes observed during this session.

        Args:
            op: The list namespaces operation.

        Returns:
            A list of namespace tuples.
        """
        namespaces = [
            _scope_to_namespace(s, self._scope_delimiter)
            for s in sorted(self._known_scopes)
        ]

        if op.match_conditions:
            namespaces = [
                ns
                for ns in namespaces
                if all(
                    _does_match(cond, ns) for cond in op.match_conditions
                )
            ]

        if op.max_depth is not None:
            namespaces = sorted({ns[: op.max_depth] for ns in namespaces})
        else:
            namespaces = sorted(namespaces)

        return namespaces[op.offset : op.offset + op.limit]

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------

    async def _ahandle_get(self, op: GetOp) -> Optional[Item]:
        """Async counterpart of :meth:`_handle_get`.

        Args:
            op: The get operation.

        Returns:
            The matching :class:`Item` or ``None``.
        """
        from azure.ai.projects.models import (
            MemorySearchOptions,
            ResponsesUserMessageItemParam,
        )

        scope = _namespace_to_scope(op.namespace, self._scope_delimiter)
        self._known_scopes.add(scope)
        try:
            result = await self._async_client.memory_stores.search_memories(
                name=self._memory_store_name,
                scope=scope,
                items=[ResponsesUserMessageItemParam(content="Retrieve all memories")],
                options=MemorySearchOptions(max_memories=100),
            )
        except Exception:
            logger.debug("Failed to search memories for get op", exc_info=True)
            return None

        for search_item in result.memories:
            mem = search_item.memory_item
            if mem.memory_id == op.key:
                namespace = _scope_to_namespace(
                    mem.scope, self._scope_delimiter
                )
                return _memory_item_to_store_item(mem, namespace)
        return None

    async def _ahandle_search(self, op: SearchOp) -> list[SearchItem]:
        """Async counterpart of :meth:`_handle_search`.

        Args:
            op: The search operation.

        Returns:
            A list of matching :class:`SearchItem` objects.
        """
        from azure.ai.projects.models import (
            MemorySearchOptions,
            ResponsesUserMessageItemParam,
        )

        scope = _namespace_to_scope(op.namespace_prefix, self._scope_delimiter)
        self._known_scopes.add(scope)

        query_text = op.query or "Retrieve all memories"
        items = [ResponsesUserMessageItemParam(content=query_text)]

        try:
            result = await self._async_client.memory_stores.search_memories(
                name=self._memory_store_name,
                scope=scope,
                items=items,
                options=MemorySearchOptions(max_memories=op.limit + op.offset),
            )
        except Exception:
            logger.debug("Failed to search memories", exc_info=True)
            return []

        search_items: list[SearchItem] = []
        for search_result in result.memories:
            mem = search_result.memory_item
            namespace = _scope_to_namespace(mem.scope, self._scope_delimiter)

            if op.filter:
                value = {
                    "content": mem.content,
                    "kind": str(mem.kind) if mem.kind else None,
                }
                if not all(value.get(k) == v for k, v in op.filter.items()):
                    continue

            search_items.append(
                _memory_item_to_search_item(mem, namespace)
            )

        return search_items[op.offset : op.offset + op.limit]

    async def _ahandle_put(self, op: PutOp) -> None:
        """Async counterpart of :meth:`_handle_put`.

        Args:
            op: The put operation.
        """
        scope = _namespace_to_scope(op.namespace, self._scope_delimiter)
        self._known_scopes.add(scope)

        if op.value is None:
            try:
                await self._async_client.memory_stores.delete_scope(
                    name=self._memory_store_name,
                    scope=scope,
                )
            except Exception:
                logger.debug("Failed to delete scope %s", scope, exc_info=True)
            self._known_scopes.discard(scope)
            return None

        items = _build_message_items(op.value)
        if not items:
            return None

        try:
            poller = await self._async_client.memory_stores.begin_update_memories(
                name=self._memory_store_name,
                scope=scope,
                items=items,
                update_delay=0,
            )
            await poller.result()
        except Exception:
            logger.debug(
                "Failed to update memories for scope %s", scope, exc_info=True
            )
        return None


def _does_match(match_condition: MatchCondition, key: tuple[str, ...]) -> bool:
    """Check whether *key* satisfies *match_condition*.

    Args:
        match_condition: The match condition to evaluate.
        key: A namespace tuple.

    Returns:
        ``True`` if the namespace matches.
    """
    match_type = match_condition.match_type
    path = match_condition.path

    if len(key) < len(path):
        return False

    if match_type == "prefix":
        for k_elem, p_elem in zip(key, path, strict=False):
            if p_elem == "*":
                continue
            if k_elem != p_elem:
                return False
        return True
    elif match_type == "suffix":
        for k_elem, p_elem in zip(reversed(key), reversed(path), strict=False):
            if p_elem == "*":
                continue
            if k_elem != p_elem:
                return False
        return True
    else:
        raise ValueError(f"Unsupported match type: {match_type}")
