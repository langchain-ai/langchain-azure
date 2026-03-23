"""Unit tests for _table_utils helpers."""

import base64

from langchain_azure_storage._table_utils import (
    chunk_data,
    escape_key,
    make_checkpoint_row_key,
    make_writes_row_key,
    parse_checkpoint_row_key,
    parse_writes_row_key,
    reassemble_data,
    unescape_key,
)


class TestKeyEscaping:
    def test_escape_roundtrip(self) -> None:
        original = "ns/with\\special#chars?here"
        escaped = escape_key(original)
        assert "/" not in escaped
        assert "\\" not in escaped
        assert "#" not in escaped
        assert "?" not in escaped
        assert unescape_key(escaped) == original

    def test_escape_empty(self) -> None:
        assert escape_key("") == ""
        assert unescape_key("") == ""

    def test_escape_no_special(self) -> None:
        val = "simple-string"
        assert escape_key(val) == val
        assert unescape_key(val) == val


class TestCheckpointRowKey:
    def test_roundtrip(self) -> None:
        ns = "my/namespace"
        cid = "1ef4f797-8335-6428-8001-8a1503f9b875"
        row_key = make_checkpoint_row_key(ns, cid)
        parsed_ns, parsed_cid = parse_checkpoint_row_key(row_key)
        assert parsed_ns == ns
        assert parsed_cid == cid

    def test_empty_namespace(self) -> None:
        ns = ""
        cid = "abc-123"
        row_key = make_checkpoint_row_key(ns, cid)
        parsed_ns, parsed_cid = parse_checkpoint_row_key(row_key)
        assert parsed_ns == ""
        assert parsed_cid == cid


class TestWritesRowKey:
    def test_roundtrip(self) -> None:
        ns = "ns"
        cid = "checkpoint-id"
        task = "task-1"
        idx = 5
        row_key = make_writes_row_key(ns, cid, task, idx)
        parsed = parse_writes_row_key(row_key)
        assert parsed == (ns, cid, task, idx)

    def test_special_chars(self) -> None:
        ns = "ns/with/slashes"
        cid = "id#hash"
        task = "task?q"
        idx = -1
        row_key = make_writes_row_key(ns, cid, task, idx)
        parsed = parse_writes_row_key(row_key)
        assert parsed == (ns, cid, task, idx)


class TestChunking:
    def test_empty_data(self) -> None:
        chunks = chunk_data(b"")
        assert chunks == {"chunk_count": "0"}
        assert reassemble_data(chunks) == b""

    def test_small_data(self) -> None:
        data = b"hello world"
        chunks = chunk_data(data)
        assert int(chunks["chunk_count"]) == 1
        assert reassemble_data(chunks) == data

    def test_large_data_roundtrip(self) -> None:
        # Create data larger than the chunk size (48 KB)
        data = b"x" * (48 * 1024 * 3 + 100)
        chunks = chunk_data(data)
        assert int(chunks["chunk_count"]) == 4
        assert reassemble_data(chunks) == data

    def test_chunks_are_base64(self) -> None:
        data = b"\x00\x01\x02\xff"
        chunks = chunk_data(data)
        # Each chunk should be valid base64
        encoded = chunks["data_0"]
        decoded = base64.b64decode(encoded)
        assert decoded == data

    def test_exact_chunk_boundary(self) -> None:
        chunk_size = 48 * 1024
        data = b"a" * (chunk_size * 2)
        chunks = chunk_data(data)
        assert int(chunks["chunk_count"]) == 2
        assert reassemble_data(chunks) == data
