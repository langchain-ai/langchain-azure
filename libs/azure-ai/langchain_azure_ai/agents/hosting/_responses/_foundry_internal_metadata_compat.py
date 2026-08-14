"""Temporary compatibility for Foundry response internal-metadata storage."""

from __future__ import annotations

import json
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, cast

from azure.ai.agentserver.responses import (
    ResponseEventStream,
    ResponseObject,
)

_RESERVED_KEY = "_internal_metadata"


@contextmanager
def encode_internal_metadata_for_checkpoint(
    stream: ResponseEventStream,
) -> Iterator[None]:
    """Encode the live metadata bag only while its checkpoint is persisted."""
    metadata = stream.response.get("metadata")
    if not isinstance(metadata, MutableMapping):
        yield
        return
    internal_metadata = metadata.get(_RESERVED_KEY)
    if not isinstance(internal_metadata, dict):
        yield
        return

    metadata[_RESERVED_KEY] = json.dumps(
        internal_metadata,
        separators=(",", ":"),
    )
    try:
        yield
    finally:
        metadata[_RESERVED_KEY] = internal_metadata


def decode_internal_metadata_from_persisted_response(
    response: ResponseObject,
) -> ResponseObject:
    """Decode Foundry's string form before seeding a recovered stream."""
    metadata = response.get("metadata")
    if not isinstance(metadata, MutableMapping):
        return response
    internal_metadata = metadata.get(_RESERVED_KEY)
    if not isinstance(internal_metadata, str):
        return response
    try:
        decoded = json.loads(internal_metadata)
    except json.JSONDecodeError:
        return response
    if not isinstance(decoded, dict):
        return response

    decoded_response = deepcopy(response)
    decoded_metadata = decoded_response.get("metadata")
    if isinstance(decoded_metadata, MutableMapping):
        decoded_metadata[_RESERVED_KEY] = cast(dict[str, Any], decoded)
    return decoded_response
