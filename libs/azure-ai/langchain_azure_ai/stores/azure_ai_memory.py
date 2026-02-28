"""Azure AI Memory Store for LangGraph."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

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

if TYPE_CHECKING:
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import MemoryStoreSearchResult

logger = logging.getLogger(__name__)

_NAMESPACE_SEP = "_"

# Matches the natural-language storage format used by _make_message:
#   The value for key 'some-key' is: {"field": "value"}
_KEY_VALUE_RE = re.compile(r"The value for key '([^']+)' is: (.+)$", re.DOTALL)


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

        # Retrieve it directly
        item = store.get(("users", "alice"), "preferences")

        # Search for relevant memories
        results = store.search(("users", "alice"), query="user preferences")
        ```

    Note:
        This store maintains an **in-process cache** for reliable key-value
        operations (``get``, ``list_namespaces``).  The cache is also used to
        return correctly-structured ``SearchItem`` objects when semantic search
        results cannot be parsed back to the original key/value format (which
        happens because the Azure AI service uses an LLM to extract and
        reformat memories from the conversation messages you supply).

        Data stored before the current process started can still be retrieved
        via Azure AI semantic search on a best-effort basis.

    Note:
        Azure AI scopes only allow characters in ``[A-Za-z0-9_-]``. Namespace
        components are joined with ``"_"``, so each component must only contain
        alphanumeric characters and hyphens (``-``).

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
        if not hasattr(project_client, "beta") or not hasattr(
            project_client.beta, "memory_stores"
        ):
            raise ValueError(
                "The provided AIProjectClient does not support the memory stores API. "
                "azure-ai-projects>=2.0.0b4 is required. "
                "Install with: pip install 'azure-ai-projects>=2.0.0b4' --pre"
            )
        self._client = project_client
        self._memory_store_name = memory_store_name
        # In-process cache: namespace → {key: {"value", "created_at", "updated_at"}}
        self._cache: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}

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
        """Serialize a key-value pair as an Azure AI conversation message.

        Uses a natural-language format so the Azure AI memory service's LLM
        can better identify and preserve the key name when extracting memories.
        """
        content = f"The value for key '{key}' is: {json.dumps(value)}"
        return {"role": "user", "type": "message", "content": content}

    def _parse_memory_content(
        self, content: str
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Try to parse a memory content string as a structured key-value pair.

        Handles two formats:

        * **Current format** – ``"The value for key 'X' is: {...}"``
        * **Legacy format** – raw JSON ``{"key": "X", "value": {...}}``

        Returns:
            A ``(key, value)`` tuple if parsing succeeds, otherwise
            ``(None, None)``.
        """
        # Current format: "The value for key 'X' is: {...}"
        m = _KEY_VALUE_RE.match(content)
        if m:
            try:
                value = json.loads(m.group(2))
                if isinstance(value, dict):
                    return m.group(1), value
            except (json.JSONDecodeError, ValueError):
                pass
        # Legacy format: {"key": ..., "value": ...}
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "key" in data and "value" in data:
                return str(data["key"]), data["value"]
        except (json.JSONDecodeError, ValueError):
            pass
        return None, None

    def _matches_query(
        self, key: str, value: dict[str, Any], query: str
    ) -> bool:
        """Check whether a key-value pair matches ``query`` via keyword search.

        Splits ``query`` into tokens of three or more characters and returns
        ``True`` if at least one token appears in the combined text of the key
        and the JSON-serialised value.
        """
        text = (key + " " + json.dumps(value)).lower()
        words = [w for w in re.split(r"\W+", query.lower()) if len(w) >= 3]
        return not words or any(w in text for w in words)

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
                # Skip AI-extracted memory summaries that cannot be parsed
                # back to the original user key-value structure.  These are
                # internally generated by the Azure AI service (UUID keys,
                # natural-language summaries) and must not be surfaced as
                # user-visible search results.
                continue

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
            # Delete the specific key from the local cache.
            ns_entries = self._cache.get(op.namespace, {})
            ns_entries.pop(op.key, None)
            if not ns_entries:
                self._cache.pop(op.namespace, None)

            # Azure AI only supports scope-level deletion.  Delete the scope
            # and re-add any remaining keys so that semantic search stays in
            # sync.
            try:
                self._client.beta.memory_stores.delete_scope(
                    name=self._memory_store_name,
                    scope=scope,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("delete_scope failed (scope=%s): %s", scope, exc)

            for key, entry in ns_entries.items():
                message = self._make_message(key, entry["value"])
                poller = self._client.beta.memory_stores.begin_update_memories(
                    name=self._memory_store_name,
                    scope=scope,
                    items=[message],
                    update_delay=0,
                )
                poller.result()
            return

        # Upsert into the local cache, preserving the original created_at.
        now = datetime.now(timezone.utc)
        existing = self._cache.get(op.namespace, {}).get(op.key, {})
        created_at: datetime = existing.get("created_at", now)
        self._cache.setdefault(op.namespace, {})[op.key] = {
            "value": op.value,
            "created_at": created_at,
            "updated_at": now,
        }

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
        # Fast path: serve from the in-process cache.
        entry = self._cache.get(op.namespace, {}).get(op.key)
        if entry is not None:
            return Item(
                value=entry["value"],
                key=op.key,
                namespace=op.namespace,
                created_at=entry["created_at"],
                updated_at=entry["updated_at"],
            )

        # Slow path: fall back to Azure AI semantic search (cross-session data).
        from azure.ai.projects.models import MemorySearchOptions

        scope = self._namespace_to_scope(op.namespace)
        query_message = {
            "role": "user",
            "type": "message",
            "content": f"The value for key '{op.key}'",
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

        # --- Local-cache results (correct key-value structure) ---
        cache_items: list[SearchItem] = []
        for ns, entries in self._cache.items():
            if ns[: len(op.namespace_prefix)] != op.namespace_prefix:
                continue
            for key, entry in entries.items():
                value = entry["value"]
                if op.filter and not all(
                    value.get(k) == v for k, v in op.filter.items()
                ):
                    continue
                if op.query and not self._matches_query(key, value, op.query):
                    continue
                cache_items.append(
                    SearchItem(
                        namespace=ns,
                        key=key,
                        value=value,
                        created_at=entry["created_at"],
                        updated_at=entry["updated_at"],
                    )
                )

        # --- Azure AI semantic-search results (may include cross-session data) ---
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
        ai_items = self._search_result_to_items(
            result,
            namespace_prefix=op.namespace_prefix,
            limit=op.limit + op.offset,
            offset=0,
            filter=op.filter,
        )

        # Merge: prefer cache items; add AI items whose keys are not already
        # present in the cache results (they may carry cross-session data).
        cache_keys = {(item.namespace, item.key) for item in cache_items}
        for ai_item in ai_items:
            if (ai_item.namespace, ai_item.key) not in cache_keys:
                cache_items.append(ai_item)
                cache_keys.add((ai_item.namespace, ai_item.key))

        return cache_items[op.offset : op.offset + op.limit]

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

    @staticmethod
    def _namespace_matches_condition(
        ns: tuple[str, ...], cond: MatchCondition
    ) -> bool:
        """Return True if ``ns`` satisfies the namespace ``MatchCondition``."""
        path = tuple(cond.path)
        if cond.match_type == "prefix":
            return ns[: len(path)] == path
        if cond.match_type == "suffix":
            return ns[-len(path) :] == path if path else True
        return True

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Handle a ``ListNamespacesOp`` using the local in-process cache."""
        namespaces: list[tuple[str, ...]] = list(self._cache.keys())

        if op.match_conditions:
            namespaces = [
                ns
                for ns in namespaces
                if all(
                    self._namespace_matches_condition(ns, c)
                    for c in op.match_conditions
                )
            ]

        if op.max_depth is not None:
            seen: set[tuple[str, ...]] = set()
            truncated: list[tuple[str, ...]] = []
            for ns in namespaces:
                t = ns[: op.max_depth]
                if t not in seen:
                    seen.add(t)
                    truncated.append(t)
            namespaces = truncated

        return namespaces[op.offset : op.offset + op.limit]
