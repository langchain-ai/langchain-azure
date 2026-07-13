# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

from langchain_azure_ai.agents.hosting._conversation_tree import (
    ConversationTreeStore,
)


def test_previous_response_resolves_root_and_exact_checkpoint(tmp_path: Path) -> None:
    store = ConversationTreeStore(tmp_path / "responses_mapping")

    root = store.get_conv_tree_metadata(
        response_id="resp-root",
        conversation_id=None,
        previous_response_id=None,
        is_recovery=False,
        load_checkpoint=True,
    )
    store.record_checkpoint("resp-root", "checkpoint-root")
    child = store.get_conv_tree_metadata(
        response_id="resp-child",
        conversation_id=None,
        previous_response_id="resp-root",
        is_recovery=False,
        load_checkpoint=True,
    )

    assert root.root_response_id == "resp-root"
    assert root.checkpoint_id is None
    assert child.root_response_id == "resp-root"
    assert child.checkpoint_id == "checkpoint-root"
    assert (store.responses_mapping_dir / "resp-child").read_text(
        encoding="utf-8"
    ) == "resp-root"


def test_conversation_uses_first_response_root_and_latest_checkpoint(
    tmp_path: Path,
) -> None:
    store = ConversationTreeStore(tmp_path / "responses_mapping")

    first = store.get_conv_tree_metadata(
        response_id="resp-first",
        conversation_id="conv-1",
        previous_response_id=None,
        is_recovery=False,
        load_checkpoint=True,
    )
    store.record_checkpoint("resp-second", "checkpoint-second")
    second = store.get_conv_tree_metadata(
        response_id="resp-second",
        conversation_id="conv-1",
        previous_response_id=None,
        is_recovery=True,
        load_checkpoint=True,
    )

    assert first.root_response_id == "resp-first"
    assert second.root_response_id == "resp-first"
    assert first.checkpoint_id is None
    assert second.checkpoint_id is None
    assert (store.conversations_mapping_dir / "conv-1").read_text(
        encoding="utf-8"
    ) == "resp-first"
    assert (store.responses_mapping_dir / "resp-second").read_text(
        encoding="utf-8"
    ) == "resp-first"
