# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Durable conversation lineage for Responses-hosted LangGraph applications.

The store connects Responses API identities to LangGraph checkpoint identities
using three directories of extensionless text files. Each file name is a lookup
key and its contents are the mapped value::

    responses_mapping/<response_id>       -> <root_response_id>
    conversations_mapping/<conversation_id> -> <root_response_id>
    checkpoints_mapping/<response_id>     -> <checkpoint_id>

These direct-path mappings let a request resolve its LangGraph thread and an
optional historical checkpoint without walking the response ancestry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConversationTreeMetadata:
    """LangGraph identity resolved for one Responses request.

    Attributes:
        root_response_id: Response ID used as LangGraph's ``thread_id`` for
            every turn and fork in this conversation tree.
        checkpoint_id: Exact LangGraph checkpoint from which this request must
            run. ``None`` means that no historical checkpoint is selected, so
            LangGraph uses the latest state for ``root_response_id`` (or starts
            a new thread when no state exists).
    """

    root_response_id: str
    checkpoint_id: str | None = None


class ConversationTreeStore:
    """Resolve and persist Responses-to-LangGraph conversation metadata.

    A response tree has one stable root response ID, which is used as the
    LangGraph ``thread_id``. Each response can additionally point to the exact
    checkpoint produced by that turn. This allows ``previous_response_id`` to
    fork from historical state instead of implicitly using the thread's latest
    checkpoint.

    Args:
        responses_mapping_dir: Directory for response-to-root mappings. When
            omitted, mappings are stored under
            ``~/.langchain-azure-ai/responses_mapping``. Conversation and
            checkpoint mappings are stored in sibling directories named
            ``conversations_mapping`` and ``checkpoints_mapping``.
    """

    def __init__(self, responses_mapping_dir: str | Path | None = None) -> None:
        mapping_root = (
            Path(responses_mapping_dir)
            if responses_mapping_dir is not None
            else Path.home() / ".langchain-azure-ai" / "responses_mapping"
        )
        self.responses_mapping_dir = mapping_root
        self.conversations_mapping_dir = mapping_root.parent / "conversations_mapping"
        self.checkpoints_mapping_dir = mapping_root.parent / "checkpoints_mapping"

    def get_conv_tree_metadata(
        self,
        *,
        response_id: str,
        conversation_id: str | None,
        previous_response_id: str | None,
        is_recovery: bool,
        load_checkpoint: bool,
    ) -> ConversationTreeMetadata:
        """Get the LangGraph conversation identity for a Responses request.

        Callers use ``root_response_id`` from the returned metadata as
        LangGraph's ``configurable.thread_id``. When ``checkpoint_id`` is
        present, callers also pass it as ``configurable.checkpoint_id`` to
        select the exact state for a continuation, fork, or recovery. A
        missing ``checkpoint_id`` means LangGraph should use the latest state
        on the resolved thread. Recovery calls return no exact checkpoint so
        they resume the latest durable progress for the interrupted response.

        Args:
            response_id: ID assigned to the current Responses API response.
            conversation_id: Conversation that owns the request, or ``None``
                when the request uses response-linked conversation state.
            previous_response_id: Response from which the request continues or
                forks, or ``None`` when no parent response was supplied.
            is_recovery: Whether this call represents recovery of the current
                response after an interrupted attempt.
            load_checkpoint: Whether checkpoint metadata is required. Pass
                ``True`` for a checkpointed graph and ``False`` otherwise.

        Returns:
            Conversation metadata containing the stable root response ID and,
            when required, the exact LangGraph checkpoint ID.

        Side Effects:
            Durably associates the current response, and a new conversation
            when applicable, with the resolved root response.

        Raises:
            ValueError: If an identifier is not safe as a mapping file name, a
                required parent/root/checkpoint mapping is missing or empty,
                or an existing mapping conflicts with the resolved root.
        """
        checkpoint_id: str | None = None

        if conversation_id:
            root_response_id = self._read_mapping(
                self.conversations_mapping_dir,
                conversation_id,
                required=False,
            )
            if root_response_id is None:
                # First turn for client side using conversation_id for conversation
                root_response_id = response_id
                self._ensure_mapping(
                    self.conversations_mapping_dir,
                    conversation_id,
                    root_response_id,
                )
        elif previous_response_id:
            root_response_id = self._read_mapping(
                self.responses_mapping_dir,
                previous_response_id,
            )
            if load_checkpoint and not is_recovery:
                checkpoint_id = self._read_mapping(
                    self.checkpoints_mapping_dir,
                    previous_response_id,
                )
        else:
            # First turn for client side using previous_response_id chain for conversation
            root_response_id = response_id

        self._ensure_mapping(
            self.responses_mapping_dir,
            response_id,
            root_response_id,
        )

        return ConversationTreeMetadata(
            root_response_id=root_response_id,
            checkpoint_id=checkpoint_id,
        )

    def record_checkpoint(self, response_id: str, checkpoint_id: str) -> None:
        """Associate a response with the latest checkpoint produced by its turn.

        Args:
            response_id: Responses API response whose progress was
                checkpointed.
            checkpoint_id: LangGraph checkpoint ID that can later be selected
                for continuation, forking, or recovery.
        """
        self._write_mapping(
            self.checkpoints_mapping_dir,
            response_id,
            checkpoint_id,
        )

    @staticmethod
    def _mapping_path(directory: Path, key: str) -> Path:
        if not key or not all(
            char.isascii() and (char.isalnum() or char in "_-") for char in key
        ):
            raise ValueError(f"Invalid filesystem mapping key: {key!r}")
        return directory / key

    def _read_mapping(
        self,
        directory: Path,
        key: str,
        *,
        required: bool = True,
    ) -> str | None:
        path = self._mapping_path(directory, key)
        try:
            value = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            if not required:
                return None
            raise ValueError(f"No durable mapping exists for {key!r} at {path}.") from None
        if not value:
            raise ValueError(f"Durable mapping for {key!r} is empty.")
        return value

    def _write_mapping(self, directory: Path, key: str, value: str) -> None:
        path = self._mapping_path(directory, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def _ensure_mapping(self, directory: Path, key: str, value: str) -> None:
        existing_value = self._read_mapping(directory, key, required=False)
        if existing_value is None:
            self._write_mapping(directory, key, value)
        elif existing_value != value:
            raise ValueError(
                f"Mapping {key!r} already contains {existing_value!r}, not {value!r}."
            )
