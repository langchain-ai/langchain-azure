"""AzureAIMemoryStore: LangGraph BaseStore backed by Azure AI Memory Stores (V2).

This module provides ``AzureAIMemoryStore``, a persisted LangGraph store that
uses the Azure AI Projects SDK V2 (``azure-ai-projects>=2.0.0b4``) to store and
retrieve memories via Azure AI Foundry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MemorySearchOptions
from azure.core.credentials import TokenCredential
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
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

logger = logging.getLogger(__name__)

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

    Args:
        memory_store_name: Name of an existing Azure AI memory store.
        endpoint: Azure AI project endpoint URL.  Falls back to the
            ``AZURE_AI_PROJECT_ENDPOINT`` environment variable.
        credential: Azure credential.  Defaults to
            ``DefaultAzureCredential`` when not provided.

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
        credential: Optional[TokenCredential] = None,
    ) -> None:
        """Initialise the store.

        Args:
            memory_store_name: Name of an existing Azure AI memory store.
            endpoint: Azure AI project endpoint.  Falls back to
                ``AZURE_AI_PROJECT_ENDPOINT`` env var.
            credential: Azure credential to use.  Defaults to
                ``DefaultAzureCredential``.
        """
        resolved_endpoint = endpoint or os.environ.get(
            "AZURE_AI_PROJECT_ENDPOINT"
        )
        if not resolved_endpoint:
            raise ValueError(
                "An Azure AI project endpoint must be provided either via the "
                "`endpoint` parameter or the `AZURE_AI_PROJECT_ENDPOINT` "
                "environment variable."
            )
        self._memory_store_name = memory_store_name
        self._credential: TokenCredential = (
            credential if credential is not None else DefaultAzureCredential()
        )
        self._client = AIProjectClient(
            endpoint=resolved_endpoint,
            credential=self._credential,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _namespace_to_scope(namespace: Tuple[str, ...]) -> str:
        """Convert a namespace tuple to an Azure AI memory store scope string."""
        return ".".join(namespace)

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

        Returns ``None`` if the content does not match the expected format or
        the JSON payload cannot be decoded.
        """
        match = _KEY_VALUE_PATTERN.match(content)
        if not match:
            return None
        key = match.group(1)
        try:
            value = json.loads(match.group(2))
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None
        return key, value

    # ------------------------------------------------------------------
    # Per-operation helpers
    # ------------------------------------------------------------------

    def _do_get(self, op: GetOp) -> Optional[Item]:
        """Handle a single GetOp."""
        scope = self._namespace_to_scope(op.namespace)
        query = f"The value for key '{op.key}'"
        try:
            result = self._client.beta.memory_stores.search_memories(
                name=self._memory_store_name,
                scope=scope,
                items=[
                    {"role": "user", "content": query, "type": "message"}
                ],
                options=MemorySearchOptions(max_memories=10),
            )
        except HttpResponseError as exc:
            logger.debug(
                "Error searching memories for key '%s' in scope '%s': %s",
                op.key,
                scope,
                exc,
            )
            return None

        now = datetime.now(timezone.utc)
        for mem_item in result.memories:
            parsed = self._parse_memory_content(mem_item.memory_item.content)
            if parsed is not None:
                found_key, value = parsed
                if found_key == op.key:
                    return Item(
                        namespace=op.namespace,
                        key=op.key,
                        value=value,
                        created_at=now,
                        updated_at=now,
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

        now = datetime.now(timezone.utc)
        items: List[SearchItem] = []
        for mem_item in result.memories:
            parsed = self._parse_memory_content(mem_item.memory_item.content)
            if parsed is None:
                # Skip memories that cannot be parsed back to key-value pairs.
                continue
            key, value = parsed
            items.append(
                SearchItem(
                    namespace=op.namespace_prefix,
                    key=key,
                    value=value,
                    created_at=now,
                    updated_at=now,
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
                # Namespace listing is not supported; always return empty.
                results.append([])
            else:
                raise ValueError(f"Unsupported operation type: {type(op)}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        """Execute a batch of store operations asynchronously.

        Runs ``batch`` in a thread pool so that I/O does not block the
        event loop.

        Args:
            ops: Iterable of store operations.

        Returns:
            A list of results matching the order of *ops*.
        """
        ops_list = list(ops)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(
                executor, self.batch, ops_list
            )
