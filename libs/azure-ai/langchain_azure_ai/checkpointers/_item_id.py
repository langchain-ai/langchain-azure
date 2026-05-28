# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Item ID utilities for composite checkpoint item identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ItemType = Literal["checkpoint", "writes", "blob"]


@dataclass
class ParsedItemId:
    """Parsed components of a checkpoint item ID.

    :ivar checkpoint_ns: The checkpoint namespace.
    :ivar checkpoint_id: The checkpoint identifier.
    :ivar item_type: The type of item (``checkpoint``, ``writes``, or ``blob``).
    :ivar sub_key: Additional key for writes or blobs.
    """

    checkpoint_ns: str
    checkpoint_id: str
    item_type: ItemType
    sub_key: str


def _encode(s: str) -> str:
    """URL-safe encode a string (escape colons and percent signs)."""
    return s.replace("%", "%25").replace(":", "%3A")


def _decode(s: str) -> str:
    """Decode a URL-safe encoded string."""
    return s.replace("%3A", ":").replace("%25", "%")


def make_item_id(
    checkpoint_ns: str,
    checkpoint_id: str,
    item_type: ItemType,
    sub_key: str = "",
) -> str:
    """Create a composite item ID.

    Format: ``{checkpoint_ns}:{checkpoint_id}:{type}:{sub_key}``.
    """
    return (
        f"{_encode(checkpoint_ns)}:{_encode(checkpoint_id)}:"
        f"{item_type}:{_encode(sub_key)}"
    )


def parse_item_id(item_id: str) -> ParsedItemId:
    """Parse a composite item ID back to its components.

    :param item_id: The composite item ID to parse.
    :raises ValueError: If the item ID format is invalid.
    """
    parts = item_id.split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid item_id format: {item_id}")

    item_type = parts[2]
    if item_type not in ("checkpoint", "writes", "blob"):
        raise ValueError(f"Invalid item_type in item_id: {item_type}")

    return ParsedItemId(
        checkpoint_ns=_decode(parts[0]),
        checkpoint_id=_decode(parts[1]),
        item_type=item_type,  # type: ignore[arg-type]
        sub_key=_decode(parts[3]),
    )
