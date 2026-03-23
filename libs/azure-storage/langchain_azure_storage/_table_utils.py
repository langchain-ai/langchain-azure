"""Utilities for Azure Table Storage checkpointer."""

from __future__ import annotations

import base64
from typing import Any

# Azure Table Storage limits: 64 KB per property, 1 MB per entity.
# We use 48 KB per chunk to stay safely under the 64 KB limit after
# base64 encoding overhead (~33% expansion).
_MAX_BINARY_CHUNK_BYTES = 48 * 1024

# Characters that are invalid in Azure Table Storage PartitionKey/RowKey.
_KEY_ESCAPE_MAP = {
    "/": "_S_",
    "\\": "_B_",
    "#": "_H_",
    "?": "_Q_",
}
_KEY_UNESCAPE_MAP = {v: k for k, v in _KEY_ESCAPE_MAP.items()}

# Separator used to join checkpoint_ns and checkpoint_id in RowKey.
_ROW_KEY_SEP = "|"


def escape_key(value: str) -> str:
    """Escape characters that are invalid in Azure Table Storage keys."""
    result = value
    for char, replacement in _KEY_ESCAPE_MAP.items():
        result = result.replace(char, replacement)
    return result


def unescape_key(value: str) -> str:
    """Unescape previously escaped Azure Table Storage key characters."""
    result = value
    for replacement, char in _KEY_UNESCAPE_MAP.items():
        result = result.replace(replacement, char)
    return result


def make_checkpoint_row_key(checkpoint_ns: str, checkpoint_id: str) -> str:
    """Build a RowKey for a checkpoint entity.

    Args:
        checkpoint_ns: The checkpoint namespace.
        checkpoint_id: The checkpoint ID (UUID6, monotonically increasing).

    Returns:
        A string suitable for use as an Azure Table RowKey.
    """
    return f"{escape_key(checkpoint_ns)}{_ROW_KEY_SEP}{escape_key(checkpoint_id)}"


def parse_checkpoint_row_key(row_key: str) -> tuple[str, str]:
    """Parse a checkpoint RowKey back into (checkpoint_ns, checkpoint_id).

    Args:
        row_key: The RowKey string from an Azure Table entity.

    Returns:
        A tuple of (checkpoint_ns, checkpoint_id).
    """
    parts = row_key.split(_ROW_KEY_SEP, 1)
    return unescape_key(parts[0]), unescape_key(parts[1])


def make_writes_row_key(
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: int,
) -> str:
    """Build a RowKey for a checkpoint-writes entity.

    Args:
        checkpoint_ns: The checkpoint namespace.
        checkpoint_id: The checkpoint ID.
        task_id: The task identifier.
        idx: Write index.

    Returns:
        A string suitable for use as an Azure Table RowKey.
    """
    sep = _ROW_KEY_SEP
    return (
        f"{escape_key(checkpoint_ns)}{sep}"
        f"{escape_key(checkpoint_id)}{sep}"
        f"{escape_key(task_id)}{sep}"
        f"{idx}"
    )


def parse_writes_row_key(row_key: str) -> tuple[str, str, str, int]:
    """Parse a writes RowKey back into its components.

    Args:
        row_key: The RowKey string from an Azure Table entity.

    Returns:
        A tuple of (checkpoint_ns, checkpoint_id, task_id, idx).
    """
    parts = row_key.split(_ROW_KEY_SEP, 3)
    return (
        unescape_key(parts[0]),
        unescape_key(parts[1]),
        unescape_key(parts[2]),
        int(parts[3]),
    )


def chunk_data(data: bytes) -> dict[str, str]:
    """Split binary data into base64-encoded chunks for Azure Table Storage.

    Each chunk is stored as a separate string property so individual
    properties stay within the 64 KB entity-property limit.

    Args:
        data: The raw bytes to chunk.

    Returns:
        A dict with keys ``data_0``, ``data_1``, ... and a
        ``chunk_count`` key holding the total number of chunks.
    """
    if not data:
        return {"chunk_count": "0"}

    chunks: dict[str, str] = {}
    idx = 0
    for offset in range(0, len(data), _MAX_BINARY_CHUNK_BYTES):
        chunk = data[offset : offset + _MAX_BINARY_CHUNK_BYTES]
        chunks[f"data_{idx}"] = base64.b64encode(chunk).decode("ascii")
        idx += 1
    chunks["chunk_count"] = str(idx)
    return chunks


def reassemble_data(entity: dict[str, Any]) -> bytes:
    """Reassemble chunked data from an Azure Table entity.

    Args:
        entity: The entity dict containing ``chunk_count`` and
            ``data_0``, ``data_1``, ... fields.

    Returns:
        The original bytes.
    """
    count = int(entity.get("chunk_count", "0"))
    if count == 0:
        return b""
    parts = []
    for i in range(count):
        encoded = entity[f"data_{i}"]
        parts.append(base64.b64decode(encoded))
    return b"".join(parts)


def dumps_metadata(serde: Any, metadata: Any) -> Any:
    """Recursively serialize metadata for storage.

    Dicts are kept as dicts (with their values serialized) so that
    top-level keys remain queryable. Non-dict values are serialized
    via the serde protocol.

    Args:
        serde: A ``SerializerProtocol`` instance.
        metadata: The metadata value to serialize.

    Returns:
        The serialized representation.
    """
    if isinstance(metadata, dict):
        return {k: dumps_metadata(serde, v) for k, v in metadata.items()}
    return serde.dumps_typed(metadata)


def loads_metadata(serde: Any, metadata: Any) -> Any:
    """Recursively deserialize metadata from storage.

    Args:
        serde: A ``SerializerProtocol`` instance.
        metadata: The stored metadata value.

    Returns:
        The deserialized metadata.
    """
    if isinstance(metadata, dict):
        return {k: loads_metadata(serde, v) for k, v in metadata.items()}
    return serde.loads_typed(metadata)
