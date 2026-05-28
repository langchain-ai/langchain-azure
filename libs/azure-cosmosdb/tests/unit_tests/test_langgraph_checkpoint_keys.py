from typing import Any

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


# ---------------------------------------------------------------------------
# Checkpoint optimistic concurrency
# ---------------------------------------------------------------------------


class TestCheckpointPut:
    def _make_saver(self) -> Any:
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

    def test_put_does_not_read_before_upsert(self) -> None:
        saver = self._make_saver()
        config = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "ns",
                "checkpoint_id": None,
            }
        }
        saver.put(config, {"id": "cp1"}, {"step": 1}, {})
        saver.container.read_item.assert_not_called()
        saver.container.upsert_item.assert_called_once()
        # No optimistic-concurrency kwargs leak into the upsert call.
        upsert_args, upsert_kwargs = saver.container.upsert_item.call_args
        assert "etag" not in upsert_kwargs
        assert "match_condition" not in upsert_kwargs

    def test_put_returns_checkpoint_id(self) -> None:
        saver = self._make_saver()
        config = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "",
                "checkpoint_id": None,
            }
        }
        result = saver.put(config, {"id": "cp-new"}, {"step": 0}, {})
        assert result["configurable"]["checkpoint_id"] == "cp-new"


class TestSyncCheckpointContextManager:
    def _make_saver(self) -> Any:
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
    def _make_saver(self) -> Any:
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
        saver.container.query_items.return_value = iter(
            [
                {"id": "checkpoint$t1$$cp2"},
            ]
        )
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

    def test_get_checkpoint_key_passes_partition_key(self) -> None:
        saver = self._make_saver()
        saver.container.query_items.return_value = iter([])
        saver._get_checkpoint_key(saver.container, "t1", "ns1", None)
        call_kwargs = saver.container.query_items.call_args[1]
        assert call_kwargs["partition_key"] == "checkpoint$t1$ns1$"

    def test_get_tuple_passes_partition_key(self) -> None:
        saver = self._make_saver()
        saver.container.query_items.return_value = iter([])
        config: dict = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "",
            }
        }
        saver.get_tuple(config)
        # _get_checkpoint_key is the first query call
        call_kwargs = saver.container.query_items.call_args[1]
        assert "partition_key" in call_kwargs

    def test_list_passes_partition_key(self) -> None:
        saver = self._make_saver()
        saver.container.query_items.return_value = iter([])
        config: dict = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "",
            }
        }
        list(saver.list(config))
        call_kwargs = saver.container.query_items.call_args[1]
        assert call_kwargs["partition_key"] == "checkpoint$t1$$"

    def test_get_checkpoint_key_known_id_skips_query(self) -> None:
        saver = self._make_saver()
        key = saver._get_checkpoint_key(saver.container, "t1", "", "cp-known")
        assert key == "checkpoint$t1$$cp-known"
        saver.container.query_items.assert_not_called()

    def test_list_uses_order_by_desc(self) -> None:
        saver = self._make_saver()
        saver.container.query_items.return_value = iter([])
        config: dict = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        list(saver.list(config))
        query = saver.container.query_items.call_args[1]["query"]
        normalized = " ".join(query.upper().split())
        assert "ORDER BY C.ID DESC" in normalized

    def test_list_pushes_before_filter(self) -> None:
        saver = self._make_saver()
        saver.container.query_items.return_value = iter([])
        config: dict = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        before: dict = {
            "configurable": {
                "thread_id": "t1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-002",
            }
        }
        list(saver.list(config, before=before))
        call_kwargs = saver.container.query_items.call_args[1]
        query = call_kwargs["query"]
        params = call_kwargs["parameters"]
        param_values = [p["value"] for p in params]
        assert "before_key" in query or any("cp-002" in str(v) for v in param_values)

    def test_list_pushes_limit_as_top(self) -> None:
        saver = self._make_saver()
        saver.container.query_items.return_value = iter([])
        config: dict = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        list(saver.list(config, limit=5))
        query = saver.container.query_items.call_args[1]["query"]
        assert "TOP" in query.upper()

    def test_list_negative_limit_raises(self) -> None:
        saver = self._make_saver()
        config: dict = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        with pytest.raises(ValueError, match="positive"):
            list(saver.list(config, limit=-1))

    def test_list_zero_limit_raises(self) -> None:
        saver = self._make_saver()
        config: dict = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        with pytest.raises(ValueError, match="positive"):
            list(saver.list(config, limit=0))

    def test_load_pending_writes_sorts_by_idx(self) -> None:
        """Pending writes must be sorted by numeric idx in Python."""
        from langchain_azure_cosmosdb._langgraph_checkpoint_store import (
            _make_checkpoint_writes_key,
        )

        saver = self._make_saver()
        serde = saver.cosmos_serde

        # Create writes with idx 0, 1, 2 but return in arbitrary order
        def _make_write(idx: int) -> dict:
            key = _make_checkpoint_writes_key("t1", "", "cp-001", "task1", idx)
            pk = _make_checkpoint_writes_key("t1", "", "cp-001", "", None)
            type_, value = serde.dumps_typed(f"val-{idx}")
            return {
                "id": key,
                "partition_key": pk,
                "thread_id": "t1",
                "channel": f"ch{idx}",
                "type": type_,
                "value": value,
            }

        # Return out of numeric order (2, 0, 1)
        saver.container.query_items.return_value = iter(
            [_make_write(2), _make_write(0), _make_write(1)]
        )
        result = saver._load_pending_writes("t1", "", "cp-001")
        channels = [r[1] for r in result]
        assert channels == ["ch0", "ch1", "ch2"]


# ---------------------------------------------------------------------------
# cosmos_client_kwargs propagation
# ---------------------------------------------------------------------------


def test_cosmos_client_kwargs_forwarded() -> None:
    """CosmosDBSaverSync forwards cosmos_client_kwargs to CosmosClient."""
    from unittest.mock import MagicMock, patch

    with patch(
        "langchain_azure_cosmosdb._langgraph_checkpoint_store.CosmosClient",
    ) as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_db = MagicMock()
        mock_client.create_database_if_not_exists.return_value = mock_db
        mock_db.create_container_if_not_exists.return_value = MagicMock()

        from langchain_azure_cosmosdb import CosmosDBSaverSync

        CosmosDBSaverSync(
            database_name="db",
            container_name="ctr",
            endpoint="https://fake.documents.azure.com:443/",
            key="fakekey",
            cosmos_client_kwargs={"retry_total": 5},
        )
        _, kwargs = mock_cls.call_args
        assert kwargs.get("retry_total") == 5
