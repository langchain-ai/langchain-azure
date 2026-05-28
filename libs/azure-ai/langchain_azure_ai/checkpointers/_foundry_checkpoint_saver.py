# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""LangGraph checkpoint saver backed by Azure AI Foundry checkpoint storage."""

from __future__ import annotations

import logging
import random as _rand
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from azure.core.credentials import TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
)

from langchain_azure_ai._api.base import experimental

from ._client import FoundryCheckpointClient
from ._item_id import ParsedItemId, make_item_id, parse_item_id
from ._models import CheckpointItem, CheckpointItemId, CheckpointSession

logger = logging.getLogger(__name__)


def _encode_typed(typed: Tuple[str, bytes]) -> bytes:
    """Pack a ``(type_tag, bytes)`` tuple into a single ``bytes`` blob.

    Wire format: 4-byte big-endian length of the UTF-8 encoded ``type_tag``,
    followed by the ``type_tag`` bytes, followed by the payload bytes.
    """
    type_tag, payload = typed
    tag_bytes = type_tag.encode("utf-8")
    return len(tag_bytes).to_bytes(4, "big") + tag_bytes + payload


def _decode_typed(blob: bytes) -> Tuple[str, bytes]:
    """Inverse of :func:`_encode_typed`."""
    if len(blob) < 4:
        raise ValueError("Encoded checkpoint payload is too short")
    tag_len = int.from_bytes(blob[:4], "big")
    if len(blob) < 4 + tag_len:
        raise ValueError("Encoded checkpoint payload is truncated")
    type_tag = blob[4 : 4 + tag_len].decode("utf-8")
    payload = blob[4 + tag_len :]
    return type_tag, payload


@experimental()
class FoundryCheckpointSaver(
    BaseCheckpointSaver[str],
    AbstractAsyncContextManager["FoundryCheckpointSaver"],
):
    """LangGraph checkpoint saver backed by Azure AI Foundry checkpoint storage.

    Implements LangGraph's :class:`~langgraph.checkpoint.base.BaseCheckpointSaver`
    interface using the Foundry checkpoint storage REST API (preview).

    This saver only supports asynchronous operations. Sync methods raise
    ``NotImplementedError``.

    :param project_endpoint: The Azure AI Foundry project endpoint URL, for
        example
        ``"https://<resource>.services.ai.azure.com/api/projects/<project-id>"``.
    :type project_endpoint: str
    :param credential: Credential for authentication. Must be an async
        credential (e.g. ``azure.identity.aio.DefaultAzureCredential``).
    :type credential: ~azure.core.credentials_async.AsyncTokenCredential
    :param serde: Optional serializer protocol. Defaults to LangGraph's
        ``JsonPlusSerializer``.
    :type serde: Optional[~langgraph.checkpoint.base.SerializerProtocol]

    Example:
        .. code-block:: python

            from langchain_azure_ai.checkpointers import FoundryCheckpointSaver
            from azure.identity.aio import DefaultAzureCredential

            async with FoundryCheckpointSaver(
                project_endpoint="https://<resource>.services.ai.azure.com/api/projects/<project-id>",
                credential=DefaultAzureCredential(),
            ) as saver:
                graph = builder.compile(checkpointer=saver)
    """

    def __init__(
        self,
        project_endpoint: str,
        credential: Union[AsyncTokenCredential, TokenCredential],
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        if not isinstance(credential, AsyncTokenCredential):
            raise TypeError(
                "FoundryCheckpointSaver requires an AsyncTokenCredential. "
                "Use an async credential like DefaultAzureCredential from "
                "azure.identity.aio."
            )
        self._client = FoundryCheckpointClient(project_endpoint, credential)
        self._session_cache: set[str] = set()

    async def __aenter__(self) -> "FoundryCheckpointSaver":
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self._client.__aexit__(exc_type, exc_val, exc_tb)

    # ---- helpers --------------------------------------------------------

    async def _ensure_session(self, thread_id: str) -> None:
        """Ensure a session exists for the thread."""
        if thread_id not in self._session_cache:
            session = CheckpointSession(session_id=thread_id)
            await self._client.upsert_session(session)
            self._session_cache.add(thread_id)

    async def _get_latest_checkpoint_id(
        self, thread_id: str, checkpoint_ns: str
    ) -> Optional[str]:
        """Find the latest checkpoint ID for a thread and namespace."""
        item_ids = await self._client.list_item_ids(thread_id)

        checkpoint_ids: List[str] = []
        for item_id in item_ids:
            try:
                parsed = parse_item_id(item_id.item_id)
                if (
                    parsed.item_type == "checkpoint"
                    and parsed.checkpoint_ns == checkpoint_ns
                ):
                    checkpoint_ids.append(parsed.checkpoint_id)
            except ValueError:
                continue

        if not checkpoint_ids:
            return None
        return max(checkpoint_ids)

    async def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[Tuple[str, str, Any]]:
        """Load pending writes for a checkpoint."""
        item_ids = await self._client.list_item_ids(thread_id)
        writes: List[Tuple[str, str, Any]] = []

        for item_id in item_ids:
            try:
                parsed = parse_item_id(item_id.item_id)
                if (
                    parsed.item_type == "writes"
                    and parsed.checkpoint_ns == checkpoint_ns
                    and parsed.checkpoint_id == checkpoint_id
                ):
                    item = await self._client.read_item(item_id)
                    if item:
                        task_id, channel, value, _ = self.serde.loads_typed(
                            _decode_typed(item.data)
                        )
                        writes.append((task_id, channel, value))
            except (ValueError, TypeError):
                continue

        return writes

    async def _load_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        versions: ChannelVersions,
    ) -> Dict[str, Any]:
        """Load channel blobs for a checkpoint."""
        channel_values: Dict[str, Any] = {}

        for channel, version in versions.items():
            blob_item_id = make_item_id(
                checkpoint_ns, checkpoint_id, "blob", f"{channel}:{version}"
            )
            item_id = CheckpointItemId(session_id=thread_id, item_id=blob_item_id)
            item = await self._client.read_item(item_id)
            if item:
                type_tag, payload = _decode_typed(item.data)
                if type_tag != "empty":
                    channel_values[channel] = self.serde.loads_typed(
                        (type_tag, payload)
                    )

        return channel_values

    # ---- async API ------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronously get a checkpoint tuple by config."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        await self._ensure_session(thread_id)

        if not checkpoint_id:
            checkpoint_id = await self._get_latest_checkpoint_id(
                thread_id, checkpoint_ns
            )
            if not checkpoint_id:
                return None

        item_id_str = make_item_id(checkpoint_ns, checkpoint_id, "checkpoint")
        item = await self._client.read_item(
            CheckpointItemId(session_id=thread_id, item_id=item_id_str)
        )
        if not item:
            return None

        checkpoint_data = self.serde.loads_typed(_decode_typed(item.data))
        checkpoint: Checkpoint = checkpoint_data["checkpoint"]
        metadata: CheckpointMetadata = checkpoint_data["metadata"]

        channel_values = await self._load_blobs(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            checkpoint.get("channel_versions", {}),
        )
        checkpoint = {**checkpoint, "channel_values": channel_values}

        pending_writes = await self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )

        parent_config: Optional[RunnableConfig] = None
        if item.parent_id:
            try:
                parent_parsed = parse_item_id(item.parent_id)
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": parent_parsed.checkpoint_ns,
                        "checkpoint_id": parent_parsed.checkpoint_id,
                    }
                }
            except ValueError:
                pass

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        await self._ensure_session(thread_id)

        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        parent_item_id: Optional[str] = None
        if parent_checkpoint_id:
            parent_item_id = make_item_id(
                checkpoint_ns, parent_checkpoint_id, "checkpoint"
            )

        # Prepare checkpoint data (without channel_values - stored as blobs)
        checkpoint_copy: Dict[str, Any] = dict(checkpoint)
        channel_values: Dict[str, Any] = checkpoint_copy.pop("channel_values", {})

        checkpoint_data = _encode_typed(
            self.serde.dumps_typed(
                {"checkpoint": checkpoint_copy, "metadata": metadata}
            )
        )

        item_id_str = make_item_id(checkpoint_ns, checkpoint_id, "checkpoint")
        items: List[CheckpointItem] = [
            CheckpointItem(
                session_id=thread_id,
                item_id=item_id_str,
                data=checkpoint_data,
                parent_id=parent_item_id,
            )
        ]

        for channel, version in new_versions.items():
            if channel in channel_values:
                blob_typed = self.serde.dumps_typed(channel_values[channel])
            else:
                blob_typed = ("empty", b"")
            blob_data = _encode_typed(blob_typed)

            blob_item_id = make_item_id(
                checkpoint_ns, checkpoint_id, "blob", f"{channel}:{version}"
            )
            items.append(
                CheckpointItem(
                    session_id=thread_id,
                    item_id=blob_item_id,
                    data=blob_data,
                    parent_id=item_id_str,
                )
            )

        await self._client.create_items(items)
        logger.debug(
            "Saved checkpoint %s to Foundry session %s", checkpoint_id, thread_id
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronously store intermediate writes for a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        checkpoint_item_id = make_item_id(checkpoint_ns, checkpoint_id, "checkpoint")

        items: List[CheckpointItem] = []
        for idx, (channel, value) in enumerate(writes):
            write_data = _encode_typed(
                self.serde.dumps_typed((task_id, channel, value, task_path))
            )
            write_item_id = make_item_id(
                checkpoint_ns, checkpoint_id, "writes", f"{task_id}:{idx}"
            )
            items.append(
                CheckpointItem(
                    session_id=thread_id,
                    item_id=write_item_id,
                    data=write_data,
                    parent_id=checkpoint_item_id,
                )
            )

        if items:
            await self._client.create_items(items)
            logger.debug("Saved %d writes for checkpoint %s", len(items), checkpoint_id)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints matching filter criteria."""
        if not config:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns")

        item_ids = await self._client.list_item_ids(thread_id)

        checkpoint_items: List[Tuple[ParsedItemId, CheckpointItemId]] = []
        for item_id in item_ids:
            try:
                parsed = parse_item_id(item_id.item_id)
                if parsed.item_type == "checkpoint" and (
                    checkpoint_ns is None or parsed.checkpoint_ns == checkpoint_ns
                ):
                    checkpoint_items.append((parsed, item_id))
            except ValueError:
                continue

        # Newest first
        checkpoint_items.sort(key=lambda x: x[0].checkpoint_id, reverse=True)

        if before:
            before_id = get_checkpoint_id(before)
            if before_id:
                checkpoint_items = [
                    (p, i) for p, i in checkpoint_items if p.checkpoint_id < before_id
                ]

        if limit:
            checkpoint_items = checkpoint_items[:limit]

        for parsed, _ in checkpoint_items:
            tuple_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": parsed.checkpoint_ns,
                    "checkpoint_id": parsed.checkpoint_id,
                }
            }
            checkpoint_tuple = await self.aget_tuple(tuple_config)
            if checkpoint_tuple:
                if filter:
                    if not all(
                        checkpoint_tuple.metadata.get(k) == v for k, v in filter.items()
                    ):
                        continue
                yield checkpoint_tuple

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread."""
        await self._client.delete_session(thread_id)
        self._session_cache.discard(thread_id)
        logger.debug("Deleted session %s", thread_id)

    # ---- sync API (not supported) ---------------------------------------

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Sync version not supported - use :meth:`aget_tuple` instead."""
        raise NotImplementedError(
            "FoundryCheckpointSaver requires async usage. " "Use aget_tuple() instead."
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Sync version not supported - use :meth:`alist` instead."""
        raise NotImplementedError(
            "FoundryCheckpointSaver requires async usage. Use alist() instead."
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Sync version not supported - use :meth:`aput` instead."""
        raise NotImplementedError(
            "FoundryCheckpointSaver requires async usage. Use aput() instead."
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Sync version not supported - use :meth:`aput_writes` instead."""
        raise NotImplementedError(
            "FoundryCheckpointSaver requires async usage. " "Use aput_writes() instead."
        )

    def delete_thread(self, thread_id: str) -> None:
        """Sync version not supported - use :meth:`adelete_thread` instead."""
        raise NotImplementedError(
            "FoundryCheckpointSaver requires async usage. "
            "Use adelete_thread() instead."
        )

    def get_next_version(self, current: Optional[str], channel: None) -> str:
        """Generate the next version ID for a channel.

        Uses string versions with format ``"{counter}.{random}"``.
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = _rand.random()
        return f"{next_v:032}.{next_h:016}"
