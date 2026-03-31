"""Unit tests for cache helper functions in _cache.py."""

import json

from langchain_azure_cosmosdb.langchain._cache import (
    _dump_generations_to_json,
    _dumps_generations,
    _hash,
    _load_generations_from_json,
    _loads_generations,
)
from langchain_core.outputs import Generation

# ---------------------------------------------------------------------------
# _hash
# ---------------------------------------------------------------------------


def test_hash_consistent() -> None:
    result1 = _hash("hello")
    result2 = _hash("hello")
    assert result1 == result2
    assert len(result1) == 64  # SHA-256 hex digest length


def test_hash_different_inputs() -> None:
    assert _hash("hello") != _hash("world")


# ---------------------------------------------------------------------------
# _dump_generations_to_json / _load_generations_from_json round-trip
# ---------------------------------------------------------------------------


def test_dump_load_generations_json_roundtrip() -> None:
    gens = [Generation(text="foo"), Generation(text="bar")]
    dumped = _dump_generations_to_json(gens)
    loaded = _load_generations_from_json(dumped)
    assert len(loaded) == 2
    assert loaded[0].text == "foo"
    assert loaded[1].text == "bar"


def test_load_generations_from_json_invalid() -> None:
    import pytest

    with pytest.raises(ValueError, match="Could not decode json"):
        _load_generations_from_json("not valid json")


# ---------------------------------------------------------------------------
# _dumps_generations / _loads_generations round-trip
# ---------------------------------------------------------------------------


def test_dumps_loads_generations_roundtrip() -> None:
    gens = [Generation(text="alpha"), Generation(text="beta")]
    serialized = _dumps_generations(gens)
    deserialized = _loads_generations(serialized)
    assert deserialized is not None
    assert len(deserialized) == 2
    assert deserialized[0].text == "alpha"
    assert deserialized[1].text == "beta"


def test_loads_generations_malformed_returns_none() -> None:
    result = _loads_generations("completely invalid {{{")
    assert result is None


def test_loads_generations_legacy_format() -> None:
    gens = [Generation(text="legacy")]
    legacy_json = json.dumps([g.dict() for g in gens])
    result = _loads_generations(legacy_json)
    assert result is not None
    assert len(result) == 1
    assert result[0].text == "legacy"
