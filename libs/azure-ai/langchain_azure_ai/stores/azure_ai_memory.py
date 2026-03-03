"""AzureAIMemoryStore: LangGraph BaseStore backed by Azure AI Memory Stores (V2).

This module provides ``AzureAIMemoryStore``, a persisted LangGraph store that
uses the Azure AI Projects SDK V2 (``azure-ai-projects>=2.0.0b4``) to store and
retrieve memories via Azure AI Foundry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterable
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

# Separator used when converting namespace tuples to Azure AI scope strings.
# Azure AI scopes only allow characters in [A-Za-z0-9_-], so dots are not safe.
_NAMESPACE_SEP = "_"

# Regex pattern used to parse memory content back to (key, value) pairs.
_KEY_VALUE_PATTERN = re.compile(
    r"The value for key '(.+?)' is: (.+)", re.DOTALL
)


@experimental()
class AzureAIMemoryStore(BaseStore):
    """A persisted LangGraph ``BaseStore`` backed by Azure AI Memory Stores.

    This store uses the Azure AI Projects SDK V2
    (``azure-ai-projects>=2.0.0b4``) to persist memories in Azure AI
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
        - Round-trip fidelity depends on the AI model's ability to extract
          and reproduce the stored JSON values.

    Note:
        Azure AI scopes only allow characters in ``[A-Za-z0-9_-]``. Namespace
        components are joined with ``"_"``, so each component must only contain
        alphanumeric characters and hyphens (``-``).

    Note:
        ``azure-ai-projects >= 2.0.0b4`` is required. Install with:
        ``pip install 'azure-ai-projects>=2.0.0b4' --pre``

    Args:
        memory_store_name: Name of an existing Azure AI memory store.
        endpoint: Azure AI project endpoint URL.  Falls back to the
            ``AZURE_AI_PROJECT_ENDPOINT`` environment variable.
        credential: Azure credential.  Defaults to
            ``DefaultAzureCredential`` when not provided.
        api_version: Optional API version override for the
            ``AIProjectClient``.
        client_kwargs: Additional keyword arguments forwarded to
            ``AIProjectClient()``.

    Example:
        ```python
        from langchain_azure_ai.stores import AzureAIMemoryStore
        from azure.identity import DefaultAzureCredential

        store = AzureAIMemoryStore(
            memory_store_name="my-memory-store",
            endpoint="https://my-resource.services.ai.azure.com/api/projects/my-project",
            credential=DefaultAzureCredential(),
        )

        # Store a value
        store.put(("users", "alice"), "prefs", {"theme": "dark"})

        # Retrieve it
        item = store.get(("users", "alice"), "prefs")
        if item:
            print(item.value)  # {"theme": "dark"}

        # Semantic search
        results = store.search(("users",), query="user preferences")
        ```
    """

    def __init__(
        self,
        memory_store_name: str,
        *,
        endpoint: Optional[str] = None,
        credential: Optional["TokenCredential"] = None,
        api_version: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the store.

        Args:
            memory_store_name: Name of an existing Azure AI memory store.
            endpoint: Azure AI project endpoint.  Falls back to
                ``AZURE_AI_PROJECT_ENDPOINT`` env var.
            credential: Azure credential to use.  Defaults to
                ``DefaultAzureCredential``.
            api_version: Optional API version override for the
                ``AIProjectClient``.
            client_kwargs: Additional keyword arguments forwarded to
                ``AIProjectClient()``.
        """
        try:
            from azure.ai.projects import AIProjectClient as _AIProjectClient
        except ImportError as exc:
            raise ImportError(
                "azure-ai-projects>=2.0.0b4 is required. "
                "Install with: pip install 'azure-ai-projects>=2.0.0b4' --pre"
            ) from exc

        import os

        from azure.identity import DefaultAzureCredential

        resolved_endpoint = endpoint or os.environ.get(
            "AZURE_AI_PROJECT_ENDPOINT"
        )
        if not resolved_endpoint:
            raise ValueError(
                "An 'endpoint' must be provided, or the "
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _namespace_to_scope(namespace: Tuple[str, ...]) -> str:
        """Convert a namespace tuple to an Azure AI memory store scope string.

        Azure AI scopes only allow ``[A-Za-z0-9_-]``, so namespace
        components are joined with ``"_"``.
        """
        return _NAMESPACE_SEP.join(namespace)

    @staticmethod
    def _make_message(key: str, value: Dict[str, Any]) -> Dict[str, Any]:
        """Build a conversation message that encodes a key-value pair.

        Uses natural language so the underlying LLM preserves the key name
        when extracting memories.
        """
        content = f"The value for key '{key}' is: {json.dumps(value)}"
        return {"role": "user", "content": content, "type": "message"}

    @staticmethod
    def _parse_memory_content(
        content: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Try to parse an Azure memory content string back to a (key, value) pair.

        Handles two formats:

        1. The natural-language format written by :meth:`_make_message`:
           ``"The value for key 'X' is: {...}"``
        2. The legacy JSON format ``{"key": "X", "value": {...}}`` that the
           Azure AI service may use when extracting memories from conversation
           history.

        Returns ``None`` if the content does not match either expected format
        or the JSON payload cannot be decoded.
        """
        # Try the natural-language format first.
        match = _KEY_VALUE_PATTERN.match(content)
        if match:
            key = match.group(1)
            try:
                value = json.loads(match.group(2))
            except json.JSONDecodeError:
                return None
            if not isinstance(value, dict):
                return None
            return key, value

        # Fall back to the legacy JSON format {"key": ..., "value": ...}.
        try:
            obj = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(obj, dict):
            return None
        key = obj.get("key")
        value = obj.get("value")
        if not isinstance(key, str) or not isinstance(value, dict):
            return None
        return key, value

    # ------------------------------------------------------------------
    # Per-operation helpers
    # ------------------------------------------------------------------

    def _do_get(self, op: GetOp) -> Optional[Item]:
        """Handle a single GetOp."""
        from azure.ai.projects.models import MemorySearchOptions

        scope = self._namespace_to_scope(op.namespace)
        query = f"The value for key '{op.key}'"
        try:
            result = self._client.beta.memory_stores.search_memories(
                name=self._memory_store_name,
                scope=scope,
                items=[
                    {"role": "user", "content": query, "type": "message"}
                ],
                # Fetch up to 20 candidates to improve the chance of finding
                # the exact key when the service returns semantically similar
                # but non-matching memories.
                options=MemorySearchOptions(max_memories=20),
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
            parsed = self._parse_memory_content(memory.content)
            if parsed is not None:
                found_key, value = parsed
                if found_key == op.key:
                    timestamp = getattr(memory, "updated_at", None)
                    return Item(
                        namespace=op.namespace,
                        key=op.key,
                        value=value,
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
                logger.debug(
                    "Error deleting scope '%s': %s", scope, exc
                )
            return

        message = self._make_message(op.key, op.value)
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
                items=[
                    {"role": "user", "content": query, "type": "message"}
                ],
                options=MemorySearchOptions(
                    max_memories=op.limit + op.offset
                ),
            )
        except HttpResponseError as exc:
            logger.debug(
                "Error searching memories in scope '%s': %s", scope, exc
            )
            return []

        items: List[SearchItem] = []
        for mem_item in result.memories:
            memory = mem_item.memory_item
            parsed = self._parse_memory_content(memory.content)
            if parsed is None:
                # Skip memories that cannot be parsed back to key-value pairs.
                continue
            key, value = parsed
            timestamp = getattr(memory, "updated_at", None)
            items.append(
                SearchItem(
                    namespace=op.namespace_prefix,
                    key=key,
                    value=value,
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
        loop = asyncio.get_event_loop()
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
