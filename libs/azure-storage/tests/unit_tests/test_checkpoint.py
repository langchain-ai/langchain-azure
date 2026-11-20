"""Unit tests for the Azure Table Storage LangGraph checkpointer."""

from __future__ import annotations

import asyncio
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.core.credentials import AzureNamedKeyCredential, AzureSasCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.core.exceptions import (
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
)
from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langchain_azure_storage.checkpoint import (
    AzureTableStorageSaver,
    _checkpoint_row_key,
    _checkpoints_query,
    _get_chunked,
    _parse_checkpoint_row_key,
    _parse_writes_row_key,
    _partition_key,
    _prefix_upper_bound,
    _put_chunked,
    _sort_writes,
    _validate_key_part,
    _writes_from_entities,
    _writes_query,
    _writes_row_key,
)

# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


class TestValidateKeyPart:
    def test_allows_normal_string(self) -> None:
        _validate_key_part("thread-123", "thread_id")  # no raise

    def test_allows_empty_string(self) -> None:
        _validate_key_part("", "checkpoint_ns")  # no raise

    def test_rejects_separator(self) -> None:
        with pytest.raises(ValueError, match="separator"):
            _validate_key_part("a$b", "thread_id")

    @pytest.mark.parametrize("bad_char", ["/", "\\", "#", "?"])
    def test_rejects_table_storage_forbidden_chars(self, bad_char: str) -> None:
        with pytest.raises(ValueError, match="Table Storage key restrictions"):
            _validate_key_part(f"a{bad_char}b", "thread_id")

    def test_rejects_control_characters(self) -> None:
        with pytest.raises(ValueError, match="Table Storage key restrictions"):
            _validate_key_part("a\tb", "thread_id")


class TestKeyBuilders:
    def test_partition_key(self) -> None:
        assert _partition_key("t1", "ns1") == "t1$ns1"

    def test_partition_key_empty_ns(self) -> None:
        assert _partition_key("t1", "") == "t1$"

    def test_checkpoint_row_key_round_trip(self) -> None:
        row_key = _checkpoint_row_key("cp1")
        assert row_key == "checkpoint$cp1"
        assert _parse_checkpoint_row_key(row_key) == "cp1"

    def test_parse_checkpoint_row_key_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid checkpoint row key"):
            _parse_checkpoint_row_key("writes$cp1")

    def test_writes_row_key_with_idx_round_trip(self) -> None:
        row_key = _writes_row_key("cp1", "task1", 0)
        assert row_key == "writes$cp1$task1$0"
        assert _parse_writes_row_key(row_key) == ("cp1", "task1", "0")

    def test_writes_row_key_negative_idx(self) -> None:
        row_key = _writes_row_key("cp1", "task1", -1)
        assert row_key == "writes$cp1$task1$-1"
        assert _parse_writes_row_key(row_key) == ("cp1", "task1", "-1")

    def test_writes_row_key_without_idx(self) -> None:
        assert _writes_row_key("cp1", "task1", None) == "writes$cp1$task1"

    def test_parse_writes_row_key_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid writes row key"):
            _parse_writes_row_key("checkpoint$cp1$task1$0")


class TestChunking:
    def test_round_trip_empty_bytes(self) -> None:
        entity: dict[str, Any] = {}
        _put_chunked(entity, "checkpoint", "json", b"")
        assert entity["checkpoint_n"] == 1
        assert _get_chunked(entity, "checkpoint") == ("json", b"")

    def test_round_trip_small_value(self) -> None:
        entity: dict[str, Any] = {}
        _put_chunked(entity, "checkpoint", "json", b"hello world")
        assert entity["checkpoint_n"] == 1
        assert _get_chunked(entity, "checkpoint") == ("json", b"hello world")

    def test_round_trip_multi_chunk_value(self) -> None:
        # Larger than the 63 KiB per-chunk size, so this must split.
        data = b"x" * (63 * 1024 + 100)
        entity: dict[str, Any] = {}
        _put_chunked(entity, "checkpoint", "bytes", data)
        assert entity["checkpoint_n"] == 2
        assert _get_chunked(entity, "checkpoint") == ("bytes", data)

    def test_round_trip_exact_chunk_boundary(self) -> None:
        data = b"y" * (63 * 1024)
        entity: dict[str, Any] = {}
        _put_chunked(entity, "value", "bytes", data)
        assert entity["value_n"] == 1
        assert _get_chunked(entity, "value") == ("bytes", data)


class TestQueryBuilders:
    def test_prefix_upper_bound_bumps_last_character(self) -> None:
        assert _prefix_upper_bound("checkpoint$") == "checkpoint%"

    @pytest.mark.parametrize("suffix", ["~", "é", "~task", "￿", "zzz"])
    def test_prefix_upper_bound_covers_every_continuation(self, suffix: str) -> None:
        prefix = "writes$cp1$"
        assert prefix <= prefix + suffix < _prefix_upper_bound(prefix)

    def test_prefix_upper_bound_excludes_sibling_prefixes(self) -> None:
        assert not "writes$cp10$task" < _prefix_upper_bound("writes$cp1$")

    def test_checkpoints_query_for_one_namespace(self) -> None:
        query_filter, parameters = _checkpoints_query("t1", "ns1")
        assert "PartitionKey eq @pk" in query_filter
        assert "before" not in parameters
        assert parameters["pk"] == "t1$ns1"
        assert parameters["lo"] == "checkpoint$"
        assert parameters["hi"] == "checkpoint%"

    def test_checkpoints_query_for_empty_namespace_is_not_unscoped(self) -> None:
        _, parameters = _checkpoints_query("t1", "")
        assert parameters["pk"] == "t1$"

    def test_checkpoints_query_spans_namespaces_when_ns_is_none(self) -> None:
        query_filter, parameters = _checkpoints_query("t1", None)
        assert "PartitionKey eq" not in query_filter
        assert "pk" not in parameters
        assert parameters["plo"] == "t1$"
        assert parameters["phi"] == "t1%"

    def test_checkpoints_query_spans_threads_when_thread_is_none(self) -> None:
        query_filter, parameters = _checkpoints_query(None, None)
        assert "PartitionKey" not in query_filter
        assert set(parameters) == {"lo", "hi"}

    def test_checkpoints_query_with_checkpoint_id_and_before(self) -> None:
        query_filter, parameters = _checkpoints_query("t1", "ns1", "cp3", "cp5")
        assert "RowKey eq @id" in query_filter
        assert "RowKey lt @before" in query_filter
        assert parameters["id"] == "checkpoint$cp3"
        assert parameters["before"] == "checkpoint$cp5"

    def test_writes_query(self) -> None:
        query_filter, parameters = _writes_query("t1$ns1", "cp1")
        assert parameters["pk"] == "t1$ns1"
        assert parameters["lo"] == "writes$cp1$"
        assert parameters["hi"] == "writes$cp1%"


class TestSortWrites:
    def test_sorts_numerically_not_lexically(self) -> None:
        entities = [
            {"RowKey": _writes_row_key("cp1", "task1", 10)},
            {"RowKey": _writes_row_key("cp1", "task1", 2)},
        ]
        sorted_entities = _sort_writes(entities)
        idxs = [_parse_writes_row_key(e["RowKey"])[2] for e in sorted_entities]
        assert idxs == ["2", "10"]

    def test_sorts_negative_special_channel_idx_first(self) -> None:
        entities = [
            {"RowKey": _writes_row_key("cp1", "task1", 0)},
            {"RowKey": _writes_row_key("cp1", "task1", -1)},
        ]
        sorted_entities = _sort_writes(entities)
        idxs = [_parse_writes_row_key(e["RowKey"])[2] for e in sorted_entities]
        assert idxs == ["-1", "0"]


def _serde() -> JsonPlusSerializer:
    return JsonPlusSerializer()


class TestWritesFromEntities:
    def test_converts_and_orders(self) -> None:
        serde = _serde()
        entities = []
        for idx, channel in [(1, "chan1"), (0, "chan0")]:
            entity: dict[str, Any] = {
                "RowKey": _writes_row_key("cp1", "task1", idx),
                "channel": channel,
            }
            type_, data = serde.dumps_typed(f"value-{idx}")
            _put_chunked(entity, "value", type_, data)
            entities.append(entity)

        result = _writes_from_entities(serde, entities)
        assert result == [
            ("task1", "chan0", "value-0"),
            ("task1", "chan1", "value-1"),
        ]


# ---------------------------------------------------------------------------
# In-memory fake TableClient/AsyncTableClient
#
# Understands exactly the query/entity shapes this module generates -- not a
# general OData engine -- so tests exercise the saver's real logic
# (chunking, key building, filtering, sorting) without a live account or the
# Azurite emulator.
# ---------------------------------------------------------------------------


class _FakeSyncTable:
    def __init__(self) -> None:
        self.entities: dict[tuple[str, str], dict[str, Any]] = {}
        self.closed = False

    def create_table(self) -> None:
        pass

    def upsert_entity(self, entity: dict[str, Any], mode: Any = None) -> dict[str, Any]:
        self.entities[(entity["PartitionKey"], entity["RowKey"])] = dict(entity)
        return {}

    def create_entity(self, entity: dict[str, Any]) -> dict[str, Any]:
        key = (entity["PartitionKey"], entity["RowKey"])
        if key in self.entities:
            raise ResourceExistsError("entity already exists")
        self.entities[key] = dict(entity)
        return {}

    def get_entity(
        self, partition_key: str, row_key: str, *, select: Any = None
    ) -> dict[str, Any]:
        key = (partition_key, row_key)
        if key not in self.entities:
            raise ResourceNotFoundError("entity not found")
        return self._project(self.entities[key], select)

    def query_entities(
        self,
        query_filter: str,
        *,
        parameters: dict[str, Any] | None = None,
        select: Any = None,
        results_per_page: Any = None,
    ) -> Iterator[dict[str, Any]]:
        parameters = parameters or {}
        pk = parameters.get("pk")
        plo = parameters.get("plo")
        phi = parameters.get("phi")
        lo = parameters.get("lo")
        hi = parameters.get("hi")
        row_id = parameters.get("id")
        before = parameters.get("before")
        results = []
        for (p, r), entity in self.entities.items():
            if pk is not None and p != pk:
                continue
            if plo is not None and not p >= plo:
                continue
            if phi is not None and not p < phi:
                continue
            if lo is not None and not r >= lo:
                continue
            if hi is not None and not r < hi:
                continue
            if row_id is not None and r != row_id:
                continue
            if before is not None and not r < before:
                continue
            results.append(self._project(entity, select))
        return iter(results)

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def _project(entity: dict[str, Any], select: Any) -> dict[str, Any]:
        if not select:
            return dict(entity)
        return {k: entity[k] for k in select if k in entity}


class _FakeAsyncTable:
    """Async adapter over `_FakeSyncTable`, sharing its backing store."""

    def __init__(self, fake: _FakeSyncTable) -> None:
        self._fake = fake

    async def create_table(self) -> None:
        self._fake.create_table()

    async def upsert_entity(self, entity: dict[str, Any], mode: Any = None) -> Any:
        return self._fake.upsert_entity(entity, mode=mode)

    async def create_entity(self, entity: dict[str, Any]) -> Any:
        return self._fake.create_entity(entity)

    async def get_entity(
        self, partition_key: str, row_key: str, *, select: Any = None
    ) -> dict[str, Any]:
        return self._fake.get_entity(partition_key, row_key, select=select)

    def query_entities(self, query_filter: str, **kwargs: Any) -> Any:
        items = list(self._fake.query_entities(query_filter, **kwargs))

        async def _agen() -> Any:
            for item in items:
                yield item

        return _agen()

    async def close(self) -> None:
        self._fake.close()


def _make_sync_saver() -> tuple[AzureTableStorageSaver, _FakeSyncTable]:
    saver = AzureTableStorageSaver("https://fake.table.core.windows.net", "checkpoints")
    fake = _FakeSyncTable()
    saver._sync_table_client = fake  # type: ignore[assignment]
    return saver, fake


def _make_async_saver() -> tuple[AzureTableStorageSaver, _FakeSyncTable]:
    saver = AzureTableStorageSaver("https://fake.table.core.windows.net", "checkpoints")
    fake = _FakeSyncTable()
    saver._async_table_client = _FakeAsyncTable(fake)  # type: ignore[assignment]
    return saver, fake


def _checkpoint(cp_id: str, **channel_values: Any) -> Checkpoint:
    channel_versions: dict[str, str | int | float] = dict.fromkeys(channel_values, 1)
    return {
        "v": 1,
        "id": cp_id,
        "ts": "2024-01-01T00:00:00.000000+00:00",
        "channel_values": channel_values,
        "channel_versions": channel_versions,
        "versions_seen": {},
        "updated_channels": None,
    }


def _config(
    thread_id: str, checkpoint_ns: str = "", checkpoint_id: str | None = None
) -> RunnableConfig:
    configurable: dict[str, Any] = {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


# ---------------------------------------------------------------------------
# Sync: put / get_tuple / list / put_writes
# ---------------------------------------------------------------------------


class TestPut:
    def test_put_stores_entity_and_returns_config(self) -> None:
        saver, fake = _make_sync_saver()
        config = _config("t1", "ns1")
        result = saver.put(config, _checkpoint("cp1", step=1), {"source": "input"}, {})

        assert result["configurable"] == {
            "thread_id": "t1",
            "checkpoint_ns": "ns1",
            "checkpoint_id": "cp1",
        }
        entity = fake.entities[("t1$ns1", "checkpoint$cp1")]
        assert entity["parent_checkpoint_id"] == ""

    def test_put_records_parent_checkpoint_id(self) -> None:
        saver, fake = _make_sync_saver()
        config = _config("t1", "", "cp1")
        saver.put(config, _checkpoint("cp2", step=2), {}, {})
        entity = fake.entities[("t1$", "checkpoint$cp2")]
        assert entity["parent_checkpoint_id"] == "cp1"

    def test_put_persists_config_metadata_and_it_is_filterable(self) -> None:
        saver, _ = _make_sync_saver()
        config: RunnableConfig = {
            **_config("t1"),
            "metadata": {"user_id": "u1"},
        }
        saver.put(config, _checkpoint("00001"), {"source": "input"}, {})
        saver.put(_config("t1"), _checkpoint("00002"), {"source": "loop"}, {})

        result = saver.get_tuple(_config("t1", "", "00001"))
        assert result is not None
        assert result.metadata == {"source": "input", "user_id": "u1"}

        filtered = list(saver.list(_config("t1"), filter={"user_id": "u1"}))
        assert [r.checkpoint["id"] for r in filtered] == ["00001"]

    def test_put_rejects_forbidden_thread_id(self) -> None:
        saver, _ = _make_sync_saver()
        config = _config("t/1")
        with pytest.raises(ValueError, match="Table Storage key restrictions"):
            saver.put(config, _checkpoint("cp1"), {}, {})

    def test_large_checkpoint_is_chunked(self) -> None:
        saver, fake = _make_sync_saver()
        config = _config("t1")
        big_value = "x" * (200 * 1024)
        saver.put(config, _checkpoint("cp1", data=big_value), {}, {})
        entity = fake.entities[("t1$", "checkpoint$cp1")]
        assert entity["checkpoint_n"] > 1


class TestGetTuple:
    def test_returns_none_when_missing(self) -> None:
        saver, _ = _make_sync_saver()
        assert saver.get_tuple(_config("unknown-thread")) is None

    def test_round_trip_with_explicit_checkpoint_id(self) -> None:
        saver, _ = _make_sync_saver()
        config = _config("t1")
        saver.put(config, _checkpoint("cp1", step=1), {"source": "input"}, {})

        result = saver.get_tuple(_config("t1", "", "cp1"))
        assert result is not None
        assert result.checkpoint["id"] == "cp1"
        assert result.checkpoint["channel_values"] == {"step": 1}
        assert result.metadata["source"] == "input"
        assert result.parent_config is None

    def test_round_trip_records_parent_config(self) -> None:
        saver, _ = _make_sync_saver()
        saver.put(_config("t1"), _checkpoint("cp1"), {}, {})
        saver.put(_config("t1", "", "cp1"), _checkpoint("cp2"), {}, {})

        result = saver.get_tuple(_config("t1", "", "cp2"))
        assert result is not None
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["checkpoint_id"] == "cp1"

    def test_without_checkpoint_id_returns_latest(self) -> None:
        saver, _ = _make_sync_saver()
        for cp_id in ["00001", "00003", "00002"]:
            saver.put(_config("t1"), _checkpoint(cp_id), {}, {})

        result = saver.get_tuple(_config("t1"))
        assert result is not None
        assert result.checkpoint["id"] == "00003"

    def test_without_checkpoint_id_returns_none_when_empty(self) -> None:
        saver, _ = _make_sync_saver()
        assert saver.get_tuple(_config("t1")) is None

    def test_includes_pending_writes(self) -> None:
        saver, _ = _make_sync_saver()
        config = saver.put(_config("t1"), _checkpoint("cp1"), {}, {})
        saver.put_writes(config, [("chan0", "val0")], "task1")

        result = saver.get_tuple(_config("t1", "", "cp1"))
        assert result is not None
        assert result.pending_writes == [("task1", "chan0", "val0")]


class TestList:
    def test_returns_empty_for_no_config_on_empty_table(self) -> None:
        saver, _ = _make_sync_saver()
        assert list(saver.list(None)) == []

    def test_no_config_lists_every_thread_with_its_own_scope(self) -> None:
        saver, _ = _make_sync_saver()
        saver.put(_config("t1", "ns-a"), _checkpoint("00001"), {"step": 1}, {})
        saver.put(_config("t2"), _checkpoint("00002"), {"step": 2}, {})

        results = list(saver.list(None))

        assert [r.config["configurable"] for r in results] == [
            {"thread_id": "t2", "checkpoint_ns": "", "checkpoint_id": "00002"},
            {"thread_id": "t1", "checkpoint_ns": "ns-a", "checkpoint_id": "00001"},
        ]

    def test_no_config_still_applies_metadata_filter_and_limit(self) -> None:
        saver, _ = _make_sync_saver()
        saver.put(_config("t1"), _checkpoint("00001"), {"step": 1}, {})
        saver.put(_config("t2"), _checkpoint("00002"), {"step": 1}, {})
        saver.put(_config("t3"), _checkpoint("00003"), {"step": 2}, {})

        filtered = list(saver.list(None, filter={"step": 1}))
        assert [r.checkpoint["id"] for r in filtered] == ["00002", "00001"]
        limited = list(saver.list(None, filter={"step": 1}, limit=1))
        assert [r.checkpoint["id"] for r in limited] == ["00002"]

    def test_thread_only_config_lists_every_namespace(self) -> None:
        saver, _ = _make_sync_saver()
        saver.put(_config("t1", ""), _checkpoint("00001"), {}, {})
        saver.put(_config("t1", "sub|node:1"), _checkpoint("00002"), {}, {})
        saver.put(_config("t10", ""), _checkpoint("00003"), {}, {})

        results = list(saver.list({"configurable": {"thread_id": "t1"}}))

        assert [
            (r.config["configurable"]["checkpoint_ns"], r.checkpoint["id"])
            for r in results
        ] == [("sub|node:1", "00002"), ("", "00001")]

    def test_thread_only_config_pending_writes_use_each_entitys_namespace(
        self,
    ) -> None:
        saver, _ = _make_sync_saver()
        saver.put(_config("t1", "ns-a"), _checkpoint("cp1"), {}, {})
        saver.put_writes(_config("t1", "ns-a", "cp1"), [("chan0", "in-a")], "task1")

        (result,) = saver.list({"configurable": {"thread_id": "t1"}})

        assert result.pending_writes == [("task1", "chan0", "in-a")]

    def test_empty_namespace_config_excludes_other_namespaces(self) -> None:
        saver, _ = _make_sync_saver()
        saver.put(_config("t1", ""), _checkpoint("00001"), {}, {})
        saver.put(_config("t1", "ns-a"), _checkpoint("00002"), {}, {})

        results = list(saver.list(_config("t1", "")))

        assert [r.checkpoint["id"] for r in results] == ["00001"]

    def test_config_checkpoint_id_selects_only_that_checkpoint(self) -> None:
        saver, _ = _make_sync_saver()
        for cp_id in ["00001", "00002", "00003"]:
            saver.put(_config("t1"), _checkpoint(cp_id), {}, {})

        results = list(saver.list(_config("t1", "", "00002")))

        assert [r.checkpoint["id"] for r in results] == ["00002"]

    def test_checkpoint_id_combines_with_before(self) -> None:
        saver, _ = _make_sync_saver()
        for cp_id in ["00001", "00002", "00003"]:
            saver.put(_config("t1"), _checkpoint(cp_id), {}, {})

        results = list(
            saver.list(_config("t1", "", "00002"), before=_config("t1", "", "00002"))
        )

        assert results == []

    def test_lists_ids_that_sort_after_tilde(self) -> None:
        saver, _ = _make_sync_saver()
        for cp_id in ["00001", "~tilde", "é-checkpoint"]:
            saver.put(_config("t1"), _checkpoint(cp_id), {}, {})

        results = list(saver.list(_config("t1")))
        latest = saver.get_tuple(_config("t1"))

        assert [r.checkpoint["id"] for r in results] == [
            "é-checkpoint",
            "~tilde",
            "00001",
        ]
        assert latest is not None
        assert latest.checkpoint["id"] == "é-checkpoint"

    def test_negative_limit_raises(self) -> None:
        saver, _ = _make_sync_saver()
        with pytest.raises(ValueError, match="positive"):
            list(saver.list(_config("t1"), limit=-1))

    def test_zero_limit_raises(self) -> None:
        saver, _ = _make_sync_saver()
        with pytest.raises(ValueError, match="positive"):
            list(saver.list(_config("t1"), limit=0))

    def test_lists_most_recent_first(self) -> None:
        saver, _ = _make_sync_saver()
        for cp_id in ["00001", "00002", "00003"]:
            saver.put(_config("t1"), _checkpoint(cp_id), {}, {})

        results = list(saver.list(_config("t1")))
        assert [r.checkpoint["id"] for r in results] == ["00003", "00002", "00001"]

    def test_respects_limit(self) -> None:
        saver, _ = _make_sync_saver()
        for cp_id in ["00001", "00002", "00003"]:
            saver.put(_config("t1"), _checkpoint(cp_id), {}, {})

        results = list(saver.list(_config("t1"), limit=2))
        assert [r.checkpoint["id"] for r in results] == ["00003", "00002"]

    def test_respects_before(self) -> None:
        saver, _ = _make_sync_saver()
        for cp_id in ["00001", "00002", "00003"]:
            saver.put(_config("t1"), _checkpoint(cp_id), {}, {})

        before = _config("t1", "", "00003")
        results = list(saver.list(_config("t1"), before=before))
        assert [r.checkpoint["id"] for r in results] == ["00002", "00001"]

    def test_applies_metadata_filter(self) -> None:
        saver, _ = _make_sync_saver()
        saver.put(_config("t1"), _checkpoint("00001"), {"step": 1}, {})
        saver.put(_config("t1"), _checkpoint("00002"), {"step": 2}, {})

        results = list(saver.list(_config("t1"), filter={"step": 2}))
        assert [r.checkpoint["id"] for r in results] == ["00002"]

    def test_isolates_by_checkpoint_ns(self) -> None:
        saver, _ = _make_sync_saver()
        saver.put(_config("t1", "ns-a"), _checkpoint("00001"), {}, {})
        saver.put(_config("t1", "ns-b"), _checkpoint("00002"), {}, {})

        results = list(saver.list(_config("t1", "ns-a")))
        assert [r.checkpoint["id"] for r in results] == ["00001"]


class TestPutWrites:
    def test_upserts_special_channel_writes(self) -> None:
        saver, fake = _make_sync_saver()
        saver.put(_config("t1"), _checkpoint("cp1"), {}, {})
        config = _config("t1", "", "cp1")
        saver.put_writes(config, [("__error__", "boom")], "task1")
        saver.put_writes(config, [("__error__", "boom-again")], "task1")

        result = saver.get_tuple(config)
        assert result is not None
        # Second put_writes call overwrote the first (upsert semantics).
        assert result.pending_writes == [("task1", "__error__", "boom-again")]

    def test_normal_channel_writes_are_create_only(self) -> None:
        saver, fake = _make_sync_saver()
        saver.put(_config("t1"), _checkpoint("cp1"), {}, {})
        config = _config("t1", "", "cp1")
        saver.put_writes(config, [("chan0", "first")], "task1")
        # A second put_writes for the same (task_id, idx) must not overwrite.
        saver.put_writes(config, [("chan0", "second")], "task1")

        result = saver.get_tuple(config)
        assert result is not None
        assert result.pending_writes == [("task1", "chan0", "first")]

    def test_mixed_batch_upserts_only_the_special_channel(self) -> None:
        saver, _ = _make_sync_saver()
        saver.put(_config("t1"), _checkpoint("cp1"), {}, {})
        config = _config("t1", "", "cp1")
        saver.put_writes(config, [("chan0", "first"), ("__resume__", "old")], "task1")
        # Retry: the resume value must change, the regular write must not.
        saver.put_writes(config, [("chan0", "second"), ("__resume__", "new")], "task1")

        result = saver.get_tuple(config)
        assert result is not None
        assert result.pending_writes == [
            ("task1", "__resume__", "new"),
            ("task1", "chan0", "first"),
        ]

    def test_pending_writes_for_task_ids_that_sort_after_tilde(self) -> None:
        saver, _ = _make_sync_saver()
        saver.put(_config("t1"), _checkpoint("cp1"), {}, {})
        config = _config("t1", "", "cp1")
        saver.put_writes(config, [("chan0", "a")], "~task")
        saver.put_writes(config, [("chan0", "b")], "é-task")

        result = saver.get_tuple(config)
        assert result is not None
        assert sorted(result.pending_writes or []) == [
            ("~task", "chan0", "a"),
            ("é-task", "chan0", "b"),
        ]

    def test_multiple_writes_ordered_by_idx(self) -> None:
        saver, _ = _make_sync_saver()
        saver.put(_config("t1"), _checkpoint("cp1"), {}, {})
        config = _config("t1", "", "cp1")
        saver.put_writes(config, [("chan0", "v0"), ("chan1", "v1")], "task1")

        result = saver.get_tuple(config)
        assert result is not None
        assert result.pending_writes == [
            ("task1", "chan0", "v0"),
            ("task1", "chan1", "v1"),
        ]

    def test_rejects_forbidden_task_id(self) -> None:
        saver, _ = _make_sync_saver()
        config = _config("t1", "", "cp1")
        with pytest.raises(ValueError, match="Table Storage key restrictions"):
            saver.put_writes(config, [("chan0", "v0")], "task/1")


# ---------------------------------------------------------------------------
# Async: aput / aget_tuple / alist / aput_writes
# ---------------------------------------------------------------------------


class TestAsyncRoundTrip:
    async def test_aput_and_aget_tuple(self) -> None:
        saver, _ = _make_async_saver()
        config = _config("t1")
        result_config = await saver.aput(
            config, _checkpoint("cp1", step=1), {"source": "input"}, {}
        )
        assert result_config["configurable"]["checkpoint_id"] == "cp1"

        result = await saver.aget_tuple(_config("t1", "", "cp1"))
        assert result is not None
        assert result.checkpoint["channel_values"] == {"step": 1}
        assert result.metadata["source"] == "input"

    async def test_aget_tuple_returns_none_when_missing(self) -> None:
        saver, _ = _make_async_saver()
        assert await saver.aget_tuple(_config("unknown")) is None

    async def test_aget_tuple_without_id_returns_latest(self) -> None:
        saver, _ = _make_async_saver()
        for cp_id in ["00001", "00003", "00002"]:
            await saver.aput(_config("t1"), _checkpoint(cp_id), {}, {})

        result = await saver.aget_tuple(_config("t1"))
        assert result is not None
        assert result.checkpoint["id"] == "00003"

    async def test_alist_orders_descending_and_respects_limit(self) -> None:
        saver, _ = _make_async_saver()
        for cp_id in ["00001", "00002", "00003"]:
            await saver.aput(_config("t1"), _checkpoint(cp_id), {}, {})

        results = [r async for r in saver.alist(_config("t1"), limit=2)]
        assert [r.checkpoint["id"] for r in results] == ["00003", "00002"]

    async def test_alist_negative_limit_raises(self) -> None:
        saver, _ = _make_async_saver()
        with pytest.raises(ValueError, match="positive"):
            async for _ in saver.alist(_config("t1"), limit=-1):
                pass

    async def test_aput_writes_then_aget_tuple_includes_pending_writes(self) -> None:
        saver, _ = _make_async_saver()
        config = await saver.aput(_config("t1"), _checkpoint("cp1"), {}, {})
        await saver.aput_writes(config, [("chan0", "val0")], "task1")

        result = await saver.aget_tuple(_config("t1", "", "cp1"))
        assert result is not None
        assert result.pending_writes == [("task1", "chan0", "val0")]

    async def test_aput_persists_config_metadata(self) -> None:
        saver, _ = _make_async_saver()
        config: RunnableConfig = {**_config("t1"), "metadata": {"user_id": "u1"}}
        await saver.aput(config, _checkpoint("00001"), {"source": "input"}, {})
        await saver.aput(_config("t1"), _checkpoint("00002"), {"source": "loop"}, {})

        results = [r async for r in saver.alist(None, filter={"user_id": "u1"})]
        assert [r.checkpoint["id"] for r in results] == ["00001"]
        assert results[0].metadata == {"source": "input", "user_id": "u1"}

    async def test_alist_without_namespace_spans_namespaces_with_own_writes(
        self,
    ) -> None:
        saver, _ = _make_async_saver()
        await saver.aput(_config("t1", ""), _checkpoint("00001"), {}, {})
        await saver.aput(_config("t1", "ns-a"), _checkpoint("00002"), {}, {})
        await saver.aput_writes(
            _config("t1", "ns-a", "00002"), [("chan0", "in-a")], "task1"
        )

        results = [r async for r in saver.alist({"configurable": {"thread_id": "t1"}})]

        assert [
            (r.config["configurable"]["checkpoint_ns"], r.pending_writes)
            for r in results
        ] == [("ns-a", [("task1", "chan0", "in-a")]), ("", [])]

    async def test_alist_with_checkpoint_id_selects_only_that_checkpoint(
        self,
    ) -> None:
        saver, _ = _make_async_saver()
        for cp_id in ["00001", "00002", "00003"]:
            await saver.aput(_config("t1"), _checkpoint(cp_id), {}, {})

        results = [r async for r in saver.alist(_config("t1", "", "00002"))]

        assert [r.checkpoint["id"] for r in results] == ["00002"]

    async def test_alist_and_aget_tuple_see_ids_that_sort_after_tilde(self) -> None:
        saver, _ = _make_async_saver()
        for cp_id in ["00001", "é-checkpoint"]:
            await saver.aput(_config("t1"), _checkpoint(cp_id), {}, {})
        await saver.aput_writes(
            _config("t1", "", "é-checkpoint"), [("chan0", "v")], "~task"
        )

        results = [r async for r in saver.alist(_config("t1"))]
        latest = await saver.aget_tuple(_config("t1"))

        assert [r.checkpoint["id"] for r in results] == ["é-checkpoint", "00001"]
        assert latest is not None
        assert latest.pending_writes == [("~task", "chan0", "v")]

    async def test_aput_writes_mixed_batch_upserts_only_the_special_channel(
        self,
    ) -> None:
        saver, _ = _make_async_saver()
        await saver.aput(_config("t1"), _checkpoint("cp1"), {}, {})
        config = _config("t1", "", "cp1")
        await saver.aput_writes(
            config, [("chan0", "first"), ("__resume__", "old")], "task1"
        )
        await saver.aput_writes(
            config, [("chan0", "second"), ("__resume__", "new")], "task1"
        )

        result = await saver.aget_tuple(config)
        assert result is not None
        assert result.pending_writes == [
            ("task1", "__resume__", "new"),
            ("task1", "chan0", "first"),
        ]

    async def test_aput_writes_upsert_for_special_channels(self) -> None:
        saver, _ = _make_async_saver()
        await saver.aput(_config("t1"), _checkpoint("cp1"), {}, {})
        config = _config("t1", "", "cp1")
        await saver.aput_writes(config, [("__error__", "first")], "task1")
        await saver.aput_writes(config, [("__error__", "second")], "task1")

        result = await saver.aget_tuple(config)
        assert result is not None
        assert result.pending_writes == [("task1", "__error__", "second")]


# ---------------------------------------------------------------------------
# Lifecycle: lazy client creation, close/aclose, context managers
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_closes_client_and_owned_credential(self) -> None:
        saver, fake = _make_sync_saver()
        owned_credential = MagicMock()
        saver._sync_owned_credential = owned_credential

        saver.close()

        assert fake.closed is True
        owned_credential.close.assert_called_once()
        assert saver._sync_table_client is None
        assert saver._sync_owned_credential is None

    def test_close_is_a_no_op_when_unused(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints"
        )
        saver.close()  # must not raise

    def test_context_manager_calls_close(self) -> None:
        saver, fake = _make_sync_saver()
        with saver as entered:
            assert entered is saver
        assert fake.closed is True

    async def test_aclose_closes_client_and_owned_credential(self) -> None:
        saver, fake = _make_async_saver()
        owned_credential = AsyncMock()
        saver._async_owned_credential = owned_credential

        await saver.aclose()

        assert fake.closed is True
        owned_credential.close.assert_awaited_once()
        assert saver._async_table_client is None
        assert saver._async_owned_credential is None

    async def test_async_context_manager_calls_aclose(self) -> None:
        saver, fake = _make_async_saver()
        async with saver as entered:
            assert entered is saver
        assert fake.closed is True


class TestCredentialResolution:
    def test_sync_resolves_default_azure_credential_when_none(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints"
        )
        with patch(
            "langchain_azure_storage.checkpoint.azure.identity.DefaultAzureCredential"
        ) as mock_cls:
            mock_cls.return_value = MagicMock(spec=DefaultAzureCredential)
            credential = saver._resolve_sync_credential(None)
            mock_cls.assert_called_once()
            assert credential is mock_cls.return_value
            assert saver._sync_owned_credential is credential

    def test_sync_rejects_async_token_credential(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints"
        )
        async_credential = MagicMock(spec=AsyncTokenCredential)
        with pytest.raises(ValueError, match="AsyncTokenCredential"):
            saver._resolve_sync_credential(async_credential)

    def test_sync_passes_through_sas_credential(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints"
        )
        sas = AzureSasCredential("sig")
        assert saver._resolve_sync_credential(sas) is sas

    async def test_async_resolves_default_azure_credential_when_none(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints"
        )
        with patch(
            "langchain_azure_storage.checkpoint.azure.identity.aio.DefaultAzureCredential"
        ) as mock_cls:
            mock_cls.return_value = MagicMock(spec=AsyncDefaultAzureCredential)
            credential = await saver._resolve_async_credential(None)
            mock_cls.assert_called_once()
            assert credential is mock_cls.return_value
            assert saver._async_owned_credential is credential

    async def test_async_rejects_sync_token_credential(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints"
        )
        sync_credential = MagicMock(spec=DefaultAzureCredential)
        with pytest.raises(ValueError, match="synchronous TokenCredential"):
            await saver._resolve_async_credential(sync_credential)

    async def test_async_passes_through_named_key_credential(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints"
        )
        named_key = AzureNamedKeyCredential("account", "a" * 10)
        assert await saver._resolve_async_credential(named_key) is named_key


class TestLazyTableCreation:
    def test_get_sync_table_creates_table_and_caches(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net",
            "checkpoints",
            credential=AzureSasCredential("sig"),
        )
        with patch("langchain_azure_storage.checkpoint.TableClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            first = saver._get_sync_table()
            second = saver._get_sync_table()

            assert first is mock_client
            assert second is mock_client
            mock_cls.assert_called_once()
            mock_client.create_table.assert_called_once()

    def test_get_sync_table_ignores_already_exists(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net",
            "checkpoints",
            credential=AzureSasCredential("sig"),
        )
        with patch("langchain_azure_storage.checkpoint.TableClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.create_table.side_effect = ResourceExistsError("exists")
            mock_cls.return_value = mock_client

            table = saver._get_sync_table()
            assert table is mock_client

    def test_get_sync_table_uses_connection_string(self) -> None:
        from langchain_azure_storage._user_agent import USER_AGENT

        saver = AzureTableStorageSaver.from_connection_string(
            "fake-conn-str", "checkpoints"
        )
        with patch("langchain_azure_storage.checkpoint.TableClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.from_connection_string.return_value = mock_client

            table = saver._get_sync_table()

            mock_cls.from_connection_string.assert_called_once_with(
                "fake-conn-str", "checkpoints", user_agent=USER_AGENT
            )
            assert table is mock_client


class TestTableCreationFailure:
    """A failed initialization must not leave a client or credential open."""

    def test_sync_failure_closes_client_and_owned_credential(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints"
        )
        with (
            patch("langchain_azure_storage.checkpoint.TableClient") as mock_cls,
            patch(
                "langchain_azure_storage.checkpoint.azure.identity."
                "DefaultAzureCredential"
            ) as credential_cls,
        ):
            mock_cls.return_value.create_table.side_effect = HttpResponseError("denied")

            with pytest.raises(HttpResponseError):
                saver._get_sync_table()

            mock_cls.return_value.close.assert_called_once()
            credential_cls.return_value.close.assert_called_once()
        assert saver._sync_table_client is None
        assert saver._sync_owned_credential is None

    def test_sync_retry_after_failure_does_not_leak_first_credential(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints"
        )
        credentials = [MagicMock(), MagicMock()]
        with (
            patch("langchain_azure_storage.checkpoint.TableClient") as mock_cls,
            patch(
                "langchain_azure_storage.checkpoint.azure.identity."
                "DefaultAzureCredential",
                side_effect=credentials,
            ),
        ):
            mock_cls.return_value.create_table.side_effect = HttpResponseError("denied")
            for _ in range(2):
                with pytest.raises(HttpResponseError):
                    saver._get_sync_table()

        for credential in credentials:
            credential.close.assert_called_once()

    def test_sync_constructor_failure_closes_owned_credential(self) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints"
        )
        with (
            patch(
                "langchain_azure_storage.checkpoint.TableClient",
                side_effect=ValueError("bad endpoint"),
            ),
            patch(
                "langchain_azure_storage.checkpoint.azure.identity."
                "DefaultAzureCredential"
            ) as credential_cls,
        ):
            with pytest.raises(ValueError, match="bad endpoint"):
                saver._get_sync_table()

            credential_cls.return_value.close.assert_called_once()
        assert saver._sync_owned_credential is None

    def test_sync_failure_does_not_close_caller_supplied_credential(self) -> None:
        credential = MagicMock(spec=DefaultAzureCredential)
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints", credential=credential
        )
        with patch("langchain_azure_storage.checkpoint.TableClient") as mock_cls:
            mock_cls.return_value.create_table.side_effect = HttpResponseError("denied")
            with pytest.raises(HttpResponseError):
                saver._get_sync_table()

        mock_cls.return_value.close.assert_called_once()
        credential.close.assert_not_called()

    @pytest.mark.parametrize(
        "error", [HttpResponseError("denied"), asyncio.CancelledError()]
    )
    async def test_async_failure_closes_client_and_owned_credential(
        self, error: BaseException
    ) -> None:
        saver = AzureTableStorageSaver(
            "https://fake.table.core.windows.net", "checkpoints"
        )
        with (
            patch("langchain_azure_storage.checkpoint.AsyncTableClient") as mock_cls,
            patch(
                "langchain_azure_storage.checkpoint.azure.identity.aio."
                "DefaultAzureCredential"
            ) as credential_cls,
        ):
            client = mock_cls.return_value = AsyncMock()
            client.create_table.side_effect = error
            credential_cls.return_value = AsyncMock()

            with pytest.raises(type(error)):
                await saver._get_async_table()

            client.close.assert_awaited_once()
            credential_cls.return_value.close.assert_awaited_once()
        assert saver._async_table_client is None
        assert saver._async_owned_credential is None
