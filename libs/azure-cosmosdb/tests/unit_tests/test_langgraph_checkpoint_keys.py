import pytest
from langchain_azure_cosmosdb._langgraph_checkpoint_store import (
    _make_checkpoint_key,
    _make_checkpoint_writes_key,
    _parse_checkpoint_key,
    _parse_checkpoint_writes_key,
)


class TestMakeCheckpointKey:
    def test_basic_key(self) -> None:
        key = _make_checkpoint_key("thread1", "ns1", "cp1")
        assert key == "checkpoint$thread1$ns1$cp1"

    def test_empty_namespace(self) -> None:
        key = _make_checkpoint_key("thread1", "", "cp1")
        assert key == "checkpoint$thread1$$cp1"

    def test_empty_checkpoint_id(self) -> None:
        key = _make_checkpoint_key("thread1", "ns1", "")
        assert key == "checkpoint$thread1$ns1$"


class TestMakeCheckpointWritesKey:
    def test_with_idx(self) -> None:
        key = _make_checkpoint_writes_key("thread1", "ns1", "cp1", "task1", 0)
        assert key == "writes$thread1$ns1$cp1$task1$0"

    def test_without_idx(self) -> None:
        key = _make_checkpoint_writes_key("thread1", "ns1", "cp1", "task1", None)
        assert key == "writes$thread1$ns1$cp1$task1"


class TestParseCheckpointKey:
    def test_valid_key(self) -> None:
        result = _parse_checkpoint_key("checkpoint$thread1$ns1$cp1")
        assert result == {
            "thread_id": "thread1",
            "checkpoint_ns": "ns1",
            "checkpoint_id": "cp1",
        }

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid checkpoint key format"):
            _parse_checkpoint_key("bad$key")

    def test_wrong_namespace(self) -> None:
        with pytest.raises(
            ValueError, match="Expected checkpoint key to start with 'checkpoint'"
        ):
            _parse_checkpoint_key("writes$thread1$ns1$cp1")


class TestParseCheckpointWritesKey:
    def test_valid_key(self) -> None:
        result = _parse_checkpoint_writes_key("writes$thread1$ns1$cp1$task1$0")
        assert result == {
            "thread_id": "thread1",
            "checkpoint_ns": "ns1",
            "checkpoint_id": "cp1",
            "task_id": "task1",
            "idx": "0",
        }

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid writes key format"):
            _parse_checkpoint_writes_key("bad$key")

    def test_wrong_namespace(self) -> None:
        with pytest.raises(
            ValueError, match="Expected writes key to start with 'writes'"
        ):
            _parse_checkpoint_writes_key("checkpoint$thread1$ns1$cp1$task1$0")
