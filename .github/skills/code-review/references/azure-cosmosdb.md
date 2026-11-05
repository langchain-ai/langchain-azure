# `langchain-azure-cosmosdb` (`libs/azure-cosmosdb`)

Vector stores, semantic cache, chat message history, and LangGraph
checkpointer / store / cache over Cosmos DB. Source lives under `src/`.

Its `.github/copilot-instructions.md` predates the migration to `uv` and still
describes a Poetry workflow; the `Makefile` is the source of truth
(`uv run --frozen ...`). Do not raise findings that assume Poetry.

## Sync and async parity

`langchain_azure_cosmosdb/` and `langchain_azure_cosmosdb/aio/` are separate
implementations with mirrored names, matching the Azure separation rule in
[azure-sdk-contracts.md](azure-sdk-contracts.md). A behavior change on one side
that is not mirrored on the other is a defect in its own right — check for the
counterpart file before assuming an omission was deliberate. Note that
`_vectorstore_documentdb.py` and `_query_constructor.py` are intentionally
sync-only.

## Cosmos DB data-model rules

These are the failure modes that show up as runtime errors in production rather
than test failures:

- **Partition keys.** A cross-partition query where a single-partition query
  was intended is a cost and latency regression; a point read that omits the
  partition key fails outright. Any change to key selection changes the
  physical layout of existing containers, so it is a breaking change for
  deployed data even when the Python signature is unchanged.
- **Container and index configuration.** Vector index type and dimensions,
  full-text index configuration, and throughput settings must stay compatible
  with containers already provisioned by earlier versions. Silently
  recreating or reconfiguring a container is worse than failing.
- **Document schema.** Field names, `id` construction, and metadata layout are
  a persisted contract. Changing them without a documented migration orphans
  existing documents.
- **Request units.** Query shape, page size, and projection drive RU cost.
  Reviewing a query change means asking what it does to RU consumption, not
  only whether it returns the right rows.

The request-charge callback reports RU consumption to callers. Preserve when it
fires, what it aggregates, and that it never suppresses or delays the
underlying result; an exception raised by a user callback must not corrupt the
operation's return value.

## NoSQL and vCore / DocumentDB

The NoSQL API and the Mongo-vCore / DocumentDB APIs have different query
languages, index models, and vector-search capabilities. Keep them separate,
and do not assume a fix in one applies to the other.

## Vector and hybrid search

Apply the `VectorStore` contract in
[ecosystem-contracts.md](ecosystem-contracts.md). Cosmos-specific points:
similarity metric and its score direction must match
`_select_relevance_score_fn`; filter translation in `_query_constructor.py`
must produce parameterized queries, never string-interpolated user values; and
hybrid / full-text ranking changes alter result ordering for existing callers.

## LangGraph integrations

The checkpointer, store, and cache implement `BaseCheckpointSaver`,
`BaseStore`, and the cache protocol — the LangGraph section of
[ecosystem-contracts.md](ecosystem-contracts.md) applies in full. In addition:

- Checkpoint and store key construction is persisted state. The dedicated
  `test_langgraph_checkpoint_keys.py` and `test_langgraph_cache_keys.py` unit
  tests exist because key-format changes are silent data-compatibility breaks;
  a change to key layout that leaves those tests untouched is suspicious.
- Thread, namespace, and checkpoint-id scoping must remain isolated across
  concurrent graphs.
- TTL and eviction behavior must be preserved, including what happens when a
  cached entry expires mid-run.
