# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Dictionary storage scoped by conversation chain."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol

from azure.ai.agentserver.core.storage import (
    DEFAULT_ITEM_TTL_SECONDS,
    FoundryStateStore,
    FoundryStorageEndpoint,
)
from azure.core.credentials_async import AsyncTokenCredential

CONVERSATION_CHAIN_STORE_PREFIX = "langchain_azure_ai.agents.hosting/responses"


class ConversationChainStoreProtocol(Protocol):
    """Get and set named dictionaries within a conversation chain.

    Implementations must satisfy these requirements:

    - Persistence: after :meth:`set` succeeds, subsequent :meth:`get` calls with
        the same conversation chain ID and key return an equal, unaltered
        dictionary for the supported lifetime.
    - Isolation: treat ``(conversation_chain_id, key)`` as the record identity.
        Operations on one identity must not read or modify another identity.
    - Capacity: accept dictionaries containing at least 512 total characters
        across all keys and values.
    - Thread safety: calls for different keys may overlap.
    - Error handling: return ``None`` only when the key is absent and raise an
        exception for all other failures. If a dictionary exceeds an
        implementation limit, reject it without truncating it or modifying the
        previously stored value.

    Implementations may optionally support TTL but it is recommended to prevent
    unbounded growth. When supported, a write-sliding TTL is preferred:
    :meth:`set` renews the lifetime and :meth:`get` does not.
    """

    async def get(
        self,
        conversation_chain_id: str,
        key: str,
    ) -> dict[str, str] | None:
        """Load a dictionary, or return ``None`` when the key is absent."""
        ...

    async def set(
        self,
        conversation_chain_id: str,
        key: str,
        data: dict[str, str],
    ) -> None:
        """Atomically replace a dictionary for a conversation chain."""
        ...


class FoundryConversationChainStore:
    """Store each conversation chain in a separate ``FoundryStateStore``.

    Store creation and reuse follow ``FoundryStateStore.get_or_create`` semantics.
    ``user_isolation``, ``item_ttl_seconds``, ``description``, and ``tags`` apply
    only when the underlying store is first created. Existing stores retain
    their settings; neither mismatch policy updates them.

    By default, each read and write fetches the store properties and fails before
    item access if isolation or TTL differs from this instance's options. Set
    ``on_mismatch="ignore"`` to skip this validation and use existing settings.
    In that mode, passing ``user_isolation=True`` does not enable isolation on
    an existing non-isolated store; callers must ensure its settings are suitable.

    Args:
        credential: Optional async Azure credential.
        endpoint: Foundry project or storage endpoint.
        user_isolation: Whether the underlying state store isolates items by
            the Foundry-resolved user. Defaults to ``False``. Applied on creation
            and checked on reuse unless ``on_mismatch="ignore"``.
        item_ttl_seconds: Write-sliding item TTL fixed when each state store is
            first created. Defaults to ``DEFAULT_ITEM_TTL_SECONDS`` (30 days).
            ``-1`` disables expiration. Checked on reuse unless
            ``on_mismatch="ignore"``.
        description: Description assigned when each state store is created.
        tags: Metadata tags assigned when each state store is created.
        user_id: Delegated end-user identity for trusted callers.
        api_version: Foundry storage API version.
        on_mismatch: Policy for isolation or TTL mismatches. ``"fail"`` (default)
            raises ``ValueError`` on reads and writes. ``"ignore"`` skips the
            property check and retains the existing settings. Description and
            tags are not compared. Backend errors propagate under both policies.
        kwargs: Additional options forwarded to ``FoundryStateStore``.

    Raises:
        ValueError: ``on_mismatch`` is not ``"fail"`` or ``"ignore"``.

    Example:
        Require Foundry user isolation, rejecting incompatible stores::

            from langchain_azure_ai.agents.hosting import (
                FoundryConversationChainStore,
            )

            store = FoundryConversationChainStore(user_isolation=True)

        Explicitly retain existing store settings instead::

            store = FoundryConversationChainStore(on_mismatch="ignore")
    """

    def __init__(
        self,
        credential: AsyncTokenCredential | None = None,
        endpoint: FoundryStorageEndpoint | str | None = None,
        *,
        user_isolation: bool = False,
        item_ttl_seconds: int = DEFAULT_ITEM_TTL_SECONDS,
        description: str | None = "LangChain conversation state",
        tags: Mapping[str, str] | None = None,
        user_id: str | None = None,
        api_version: str = "v1",
        on_mismatch: Literal["fail", "ignore"] = "fail",
        **kwargs: Any,
    ) -> None:
        if on_mismatch not in ("fail", "ignore"):
            raise ValueError("on_mismatch must be 'fail' or 'ignore'")
        self._credential = credential
        self._endpoint = endpoint
        self._user_isolation = user_isolation
        self._item_ttl_seconds = item_ttl_seconds
        self._description = description
        self._tags = tags
        self._user_id = user_id
        self._api_version = api_version
        self._on_mismatch = on_mismatch
        self._kwargs = kwargs

    @staticmethod
    def _store_name(conversation_chain_id: str) -> str:
        return f"{CONVERSATION_CHAIN_STORE_PREFIX}/{conversation_chain_id}"

    async def _validate_store(self, state_store: FoundryStateStore) -> None:
        if self._on_mismatch == "ignore":
            return
        properties = await state_store.get()
        if properties.user_isolation != self._user_isolation:
            raise ValueError(
                f"State store {state_store.name!r} already exists with "
                f"user_isolation={properties.user_isolation}; expected "
                f"{self._user_isolation}"
            )
        if properties.item_ttl_seconds != self._item_ttl_seconds:
            raise ValueError(
                f"State store {state_store.name!r} already exists with "
                f"item_ttl_seconds={properties.item_ttl_seconds}; expected "
                f"{self._item_ttl_seconds}"
            )

    async def get(
        self,
        conversation_chain_id: str,
        key: str,
    ) -> dict[str, str] | None:
        """Load a dictionary using the underlying store's existing settings.

        Args:
            conversation_chain_id: Identifier of the conversation chain.
            key: Record key within the conversation chain.

        Returns:
            The stored string dictionary, or ``None`` when the key is absent.

        Raises:
            ValueError: Isolation or TTL differs and ``on_mismatch="fail"``.
            TypeError: The stored value is not a string dictionary.
        """
        state_store = await FoundryStateStore.get_or_create(
            self._store_name(conversation_chain_id),
            self._credential,
            self._endpoint,
            user_isolation=self._user_isolation,
            item_ttl_seconds=self._item_ttl_seconds,
            description=self._description,
            tags=self._tags,
            user_id=self._user_id,
            api_version=self._api_version,
            **self._kwargs,
        )
        async with state_store:
            await self._validate_store(state_store)
            item = await state_store.get_item(key)
        if item is None:
            return None
        if not isinstance(item.value, dict) or not all(
            isinstance(item_key, str) and isinstance(value, str)
            for item_key, value in item.value.items()
        ):
            raise TypeError(
                f"Conversation chain record {key!r} is not a string dictionary"
            )
        return item.value

    async def set(
        self,
        conversation_chain_id: str,
        key: str,
        data: dict[str, str],
    ) -> None:
        """Atomically replace a dictionary using the store's existing settings.

        Args:
            conversation_chain_id: Identifier of the conversation chain.
            key: Record key within the conversation chain.
            data: String dictionary to persist without alteration.

        Raises:
            ValueError: Isolation or TTL differs and ``on_mismatch="fail"``.
        """
        state_store = await FoundryStateStore.get_or_create(
            self._store_name(conversation_chain_id),
            self._credential,
            self._endpoint,
            user_isolation=self._user_isolation,
            item_ttl_seconds=self._item_ttl_seconds,
            description=self._description,
            tags=self._tags,
            user_id=self._user_id,
            api_version=self._api_version,
            **self._kwargs,
        )
        async with state_store:
            await self._validate_store(state_store)
            await state_store.set_item(key, dict(data))
