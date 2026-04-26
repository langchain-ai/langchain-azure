import pytest
from azure.core import MatchConditions
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


# ---------------------------------------------------------------------------
# Checkpoint optimistic concurrency
# ---------------------------------------------------------------------------


class TestCheckpointOptimisticConcurrency:
    def _make_saver(self):
        from unittest.mock import MagicMock

        from langchain_azure_cosmosdb._langgraph_checkpoint_store import (
            CosmosDBSaverSync,
            _CosmosSerializer,
        )
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        saver = CosmosDBSaverSync.__new__(CosmosDBSaverSync)
        saver.serde = JsonPlusSerializer()
        saver.cosmos_serde = _CosmosSerializer(saver.serde)
        saver.client = MagicMock()
        saver.container = MagicMock()
        return saver

    def test_put_reads_existing_etag(self) -> None:
        saver = self._make_saver()
        saver.container.read_item.return_value = {
            "id": "checkpoint$t1$ns$$cp1", "_etag": '"etag-existing"',
        }
        config = {
            "configurable": {
                "thread_id": "t1", "checkpoint_ns": "ns", "checkpoint_id": None,
            }
        }
        saver.put(config, {"id": "cp1"}, {"step": 1}, {})
        upsert_kwargs = saver.container.upsert_item.call_args[1]
        assert upsert_kwargs["etag"] == '"etag-existing"'
        assert upsert_kwargs["match_condition"] == MatchConditions.IfNotModified

    def test_put_new_checkpoint_upserts_without_etag(self) -> None:
        from azure.cosmos.exceptions import CosmosHttpResponseError

        saver = self._make_saver()
        saver.container.read_item.side_effect = CosmosHttpResponseError(
            status_code=404, message="Not found"
        )
        config = {
            "configurable": {
                "thread_id": "t1", "checkpoint_ns": "", "checkpoint_id": None,
            }
        }
        result = saver.put(config, {"id": "cp-new"}, {"step": 0}, {})
        assert result["configurable"]["checkpoint_id"] == "cp-new"
        saver.container.upsert_item.assert_called_once()


class TestSyncCheckpointContextManager:
    def _make_saver(self):
        from unittest.mock import MagicMock

        from langchain_azure_cosmosdb._langgraph_checkpoint_store import (
            CosmosDBSaverSync,
            _CosmosSerializer,
        )
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        saver = CosmosDBSaverSync.__new__(CosmosDBSaverSync)
        saver.serde = JsonPlusSerializer()
        saver.cosmos_serde = _CosmosSerializer(saver.serde)
        saver.client = MagicMock()
        saver.container = MagicMock()
        return saver

    def test_close(self) -> None:
        saver = self._make_saver()
        saver.close()
        saver.client.close.assert_called_once()

    def test_context_manager(self) -> None:
        saver = self._make_saver()
        with saver as s:
            assert s is saver
        saver.client.close.assert_called_once()


class TestCheckpointQueryOptimization:
    def _make_saver(self):
        from unittest.mock import MagicMock

        from langchain_azure_cosmosdb._langgraph_checkpoint_store import (
            CosmosDBSaverSync,
            _CosmosSerializer,
        )
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        saver = CosmosDBSaverSync.__new__(CosmosDBSaverSync)
        saver.serde = JsonPlusSerializer()
        saver.cosmos_serde = _CosmosSerializer(saver.serde)
        saver.client = MagicMock()
        saver.container = MagicMock()
        return saver

    def test_get_checkpoint_key_uses_top_1_order_by(self) -> None:
        saver = self._make_saver()
        saver.container.query_items.return_value = iter([
            {"id": "checkpoint$t1$$cp2"},
        ])
        key = saver._get_checkpoint_key(saver.container, "t1", "", None)
        query_arg = saver.container.query_items.call_args[1]["query"]
        assert "TOP 1" in query_arg
        assert "ORDER BY" in query_arg
        assert "DESC" in query_arg
        assert key == "checkpoint$t1$$cp2"

    def test_get_checkpoint_key_no_cross_partition(self) -> None:
        saver = self._make_saver()
        saver.container.query_items.return_value = iter([])
        saver._get_checkpoint_key(saver.container, "t1", "", None)
        call_kwargs = saver.container.query_items.call_args[1]
        assert "enable_cross_partition_query" not in call_kwargs

    def test_get_checkpoint_key_known_id_skips_query(self) -> None:
        saver = self._make_saver()
        key = saver._get_checkpoint_key(saver.container, "t1", "", "cp-known")
        assert key == "checkpoint$t1$$cp-known"
        saver.container.query_items.assert_not_called()
