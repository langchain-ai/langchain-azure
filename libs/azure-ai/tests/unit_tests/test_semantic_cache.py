"""Unit tests for AzureCosmosDBMongoVCoreSemanticCache.

``langchain_azure_ai.vectorstores.cache`` imports ``langchain_azure_cosmosdb``
at module scope, but that package isn't a declared dependency of
langchain-azure-ai and generally isn't installed in this package's dev
environment. A minimal stub is registered in ``sys.modules`` below (before
importing the cache module) so these tests can run without it, following the
same pattern used for other optional SDKs in this test suite (see
``test_speech_to_text_tool.py``).

The fake vector store below intentionally mirrors the real backend's
behaviour: every ``AzureDocumentDBVectorSearch`` created for a given
``llm_string`` shares the *same* underlying collection -- Cosmos DB Mongo
vCore's ``cosmosSearch`` index is defined over a field for the whole physical
collection, not scoped by the vector store's ``index_name`` label -- and
``similarity_search``/``delete_many`` only exclude non-matching documents
when an explicit filter says so, exactly like the real ``cosmosSearch``
``filter`` argument and MongoDB's ``delete_many``.
"""

from __future__ import annotations

import sys
from enum import Enum
from types import ModuleType
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.outputs import Generation

# ---------------------------------------------------------------------------
# Stub `langchain_azure_cosmosdb` so the cache module can be imported without
# the real package installed.
# ---------------------------------------------------------------------------


def _matches(doc: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    for dotted_key, expected in flt.items():
        node: Any = doc
        for part in dotted_key.split("."):
            if not isinstance(node, dict):
                return False
            node = node.get(part)
        if node != expected:
            return False
    return True


class FakeCollection:
    """Shared in-memory stand-in for a single pymongo Collection."""

    def __init__(self) -> None:
        self.docs: List[Dict[str, Any]] = []

    def delete_many(self, flt: Optional[Dict[str, Any]]) -> None:
        if not flt:
            self.docs = []
            return
        self.docs = [d for d in self.docs if not _matches(d, flt)]


def _make_cosmosdb_stub() -> ModuleType:
    module = ModuleType("langchain_azure_cosmosdb")

    class CosmosDBSimilarityType(str, Enum):
        COS = "COS"
        IP = "IP"
        L2 = "L2"

    class CosmosDBVectorSearchType(str, Enum):
        VECTOR_IVF = "vector-ivf"
        VECTOR_HNSW = "vector-hnsw"
        VECTOR_DISKANN = "vector-diskann"

    class CosmosDBVectorSearchCompression(str, Enum):
        PQ = "pq"
        HALF = "half"

    class AzureDocumentDBVectorSearch:
        def __init__(
            self,
            collection: FakeCollection,
            embedding: Any = None,
            *,
            index_name: str = "vectorSearchIndex",
            **kwargs: Any,
        ) -> None:
            self._collection = collection
            self._index_name = index_name

        def index_exists(self) -> bool:
            return True

        def create_index(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {}

        def get_collection(self) -> FakeCollection:
            return self._collection

        def add_texts(
            self,
            texts: List[str],
            metadatas: Optional[List[Dict[str, Any]]] = None,
            **kwargs: Any,
        ) -> List[int]:
            metadatas = metadatas or [{} for _ in texts]
            for text, metadata in zip(texts, metadatas):
                self._collection.docs.append(
                    {"textContent": text, "metadata": metadata}
                )
            return list(range(len(texts)))

        def similarity_search(
            self,
            query: str,
            k: int = 4,
            pre_filter: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> List[Document]:
            # Every stored document is treated as an equally strong
            # nearest-neighbour match: only `pre_filter` decides what comes
            # back, isolating the behaviour under test from embedding math.
            candidates = self._collection.docs
            if pre_filter:
                candidates = [d for d in candidates if _matches(d, pre_filter)]
            return [
                Document(page_content=d["textContent"], metadata=d["metadata"])
                for d in candidates[:k]
            ]

    module.AzureDocumentDBVectorSearch = AzureDocumentDBVectorSearch  # type: ignore[attr-defined]
    module.CosmosDBSimilarityType = CosmosDBSimilarityType  # type: ignore[attr-defined]
    module.CosmosDBVectorSearchType = CosmosDBVectorSearchType  # type: ignore[attr-defined]
    module.CosmosDBVectorSearchCompression = CosmosDBVectorSearchCompression  # type: ignore[attr-defined]
    return module


sys.modules.setdefault("langchain_azure_cosmosdb", _make_cosmosdb_stub())

from langchain_azure_ai.vectorstores.cache import (  # noqa: E402
    AzureCosmosDBMongoVCoreSemanticCache,
)


def _make_cache(
    shared_collection: FakeCollection,
) -> AzureCosmosDBMongoVCoreSemanticCache:
    # A dict-of-dicts stands in for `MongoClient[db_name][collection_name]`:
    # every llm_string's vector store is handed the exact same collection
    # object below, matching how the real cache shares one Cosmos DB
    # collection across every `_get_llm_cache(llm_string)` call.
    cosmosdb_client = {"db": {"coll": shared_collection}}
    return AzureCosmosDBMongoVCoreSemanticCache(
        cosmosdb_connection_string="mongodb://example",
        database_name="db",
        collection_name="coll",
        embedding=None,  # type: ignore[arg-type]
        cosmosdb_client=cosmosdb_client,
    )


def test_lookup_does_not_return_a_different_llm_strings_entry() -> None:
    """A cache entry written for one model must not be served to another."""
    collection = FakeCollection()
    cache = _make_cache(collection)

    cache.update("what is 2+2?", "model-a", [Generation(text="4")])

    # Same prompt, different llm_string (e.g. a different deployment or a
    # different temperature) -- must be a cache miss, not model-a's answer.
    assert cache.lookup("what is 2+2?", "model-b") is None

    # The original llm_string still gets its own entry back.
    assert cache.lookup("what is 2+2?", "model-a") == [Generation(text="4")]


def test_clear_only_removes_entries_for_the_given_llm_string() -> None:
    """Clearing one model's cache must not wipe out every other model's."""
    collection = FakeCollection()
    cache = _make_cache(collection)

    cache.update("what is 2+2?", "model-a", [Generation(text="4")])
    cache.update("what is 2+2?", "model-b", [Generation(text="four")])

    cache.clear(llm_string="model-a")

    assert cache.lookup("what is 2+2?", "model-a") is None
    assert cache.lookup("what is 2+2?", "model-b") == [Generation(text="four")]


def test_update_then_lookup_round_trips_for_the_same_llm_string() -> None:
    """Sanity check: the common single-model path still works after the fix."""
    collection = FakeCollection()
    cache = _make_cache(collection)

    cache.update("2+2?", "model-a", [Generation(text="4")])

    assert cache.lookup("2+2?", "model-a") == [Generation(text="4")]
