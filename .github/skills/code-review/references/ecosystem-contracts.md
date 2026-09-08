# Upstream contracts: LangChain, LangGraph, Deep Agents

Read this when a change implements, overrides, or calls into a LangChain,
LangGraph, or Deep Agents base class. These are the contract details that
integrations get wrong most often; each one is a defect a reviewer can state
concretely rather than a style preference.

Contents:

- [Standard test suites](#standard-test-suites)
- [VectorStore](#vectorstore)
- [BaseChatModel](#basechatmodel)
- [BaseTool](#basetool)
- [LangGraph persistence](#langgraph-persistence)
- [Deep Agents backends](#deep-agents-backends)

## Standard test suites

`langchain-tests` ships the conformance suites an integration is expected to
pass. When a change adds or materially alters an integration of one of these
types, the absence of the matching suite is a legitimate finding.

| Integration | Suite |
|---|---|
| Chat model | `langchain_tests.unit_tests.ChatModelUnitTests`, `langchain_tests.integration_tests.ChatModelIntegrationTests` |
| Embeddings | `EmbeddingsUnitTests`, `EmbeddingsIntegrationTests` |
| Tool | `ToolsUnitTests`, `ToolsIntegrationTests` |
| Vector store | `VectorStoreIntegrationTests` |
| Retriever | `RetrieversIntegrationTests` |
| KV store / cache | `BaseStoreSyncTests`, `BaseStoreAsyncTests`, `SyncCacheTestSuite`, `AsyncCacheTestSuite` |
| Deep Agents sandbox | `SandboxIntegrationTests` (exported only when `deepagents` is importable) |

Rules that apply to all of them:

- A standard test may not be deleted or silently overridden. Opting out
  requires `@pytest.mark.xfail(reason=...)` with a real reason; a bare override
  or `skip` violates the suite's own meta-test.
- A test class may not inherit from more than one `langchain_tests` base.
- `VectorStoreIntegrationTests` requires a `vectorstore` fixture that yields an
  **empty** store and cleans up in a `finally` block. A fixture that leaks
  state between tests produces false passes.
- LangGraph checkpointers have their own package,
  `langgraph-checkpoint-conformance` (`from langgraph.checkpoint.conformance
  import checkpointer_test, validate`). LangGraph's `BaseCache` has no
  conformance suite, so a cache implementation must bring its own tests.

## VectorStore

- Only `similarity_search` and the `from_texts` classmethod are abstract.
- **The `add_texts` / `add_documents` mutual-delegation trap:** each base
  implementation delegates to the other and only if the *other* is overridden.
  Override exactly one. Overriding neither raises `NotImplementedError` at call
  time, not at import, so it survives a green unit-test run.
- `kwargs["ids"]` takes precedence over `Document.id`. The base collects IDs
  when `any(ids)` is true, not `all(ids)`, so a partially populated list is
  passed through with `None` holes.
- These raise by default and must be overridden if advertised: `delete`,
  `similarity_search_with_score`, `_select_relevance_score_fn`,
  `max_marginal_relevance_search`, `max_marginal_relevance_search_by_vector`.
- **Score direction.** `similarity_search_with_score` returns provider-native
  scores. `similarity_search_with_relevance_scores` must return values in
  `[0, 1]` where **1 is most similar**. The base only emits a `warnings.warn`
  when a value escapes `[0, 1]`, so an inverted distance/similarity conversion
  silently reverses ranking. The built-in normalizers all assume a *distance*
  input.
- `get_by_ids(ids, /)` is positional-only, must set `Document.id`, may return
  fewer documents than requested, guarantees no ordering, and **must not raise**
  for missing IDs. This is the one place where LangChain's "do not raise" rule
  overrides Azure's "raise, never return a sentinel" rule.
- `delete` returns `bool | None`, where `None` specifically means "not
  implemented". `delete(ids=None)` means delete everything.
- `upsert` is **not** part of `VectorStore`; that contract lives on
  `langchain_core.indexing.DocumentIndex`.
- The async methods fall back to `run_in_executor` around their sync twins.
  Inheriting that fallback when the package already has an `azure.*.aio` client
  is a real defect: it burns a thread per call and serializes concurrency.

## BaseChatModel

- Only `_generate` and the `_llm_type` property are abstract. `_agenerate`
  falls back to `run_in_executor`, `_stream` raises, and `_astream` wraps
  `_stream`.
- If `_agenerate` delegates to `_generate`, it must convert the callback
  manager with `run_manager.get_sync()`.
- `usage_metadata` shape: required `input_tokens`, `output_tokens`,
  `total_tokens` (which must equal input + output), plus optional
  `input_token_details` (`audio`, `cache_creation`, `cache_read`) and
  `output_token_details` (`audio`, `reasoning`). Detail values need not sum to
  the totals, but `input_tokens` must be at least the sum of its details.
- `_get_ls_params` must return `ls_model_type="chat"`, a stable snake_case
  `ls_provider`, and `ls_model_name`. It resolves the model name as
  `kwargs["model"]`, then `self.model`, then `self.model_name` — so any class
  keyed on `azure_deployment`, `deployment_name`, or `model_id` **must**
  override it or it will report the wrong model to tracing.
- Streaming correctness, in rough order of how often it breaks:
  - the final chunk must carry `chunk_position="last"`, or `tool_call_chunks`
    never resolve into `tool_calls` on the aggregated message;
  - `ToolCallChunk.index` must be a stable, non-`None` int per tool call, since
    chunks merge only on equal non-`None` index;
  - emit `input_tokens` on exactly one chunk and `output_tokens` per chunk,
    otherwise aggregation multiply-counts input tokens;
  - propagate the provider message `id` on chunks, or the aggregate falls back
    to a synthetic `lc_*` id;
  - `response_metadata["model_name"]` must be a non-empty string.
- `with_structured_output(include_raw=True)` must capture parse failures into
  `parsing_error`; with `include_raw=False` it must raise.

## BaseTool

- Required members: `name`, `description`, and `_run`. `_arun` is optional.
- `args_schema` accepts a Pydantic model or a raw JSON-Schema dict. `_run` must
  declare real named parameters matching it.
- `run_manager` and a `RunnableConfig` parameter are injected **by parameter
  name and annotation** only when declared.
- Anything the model must not see — state, credentials, tool call IDs — must be
  annotated `InjectedToolArg`, `InjectedToolCallId`, or typed as `ToolRuntime`
  so it is stripped from `tool_call_schema`. Passing such values as ordinary
  schema fields exposes them to the model and is a security finding.
- `response_format="content_and_artifact"` requires `_run` to return a 2-tuple.
- Recoverable failures should raise `ToolException` so `handle_tool_error` can
  turn them into a `ToolMessage(status="error")` the agent can recover from. A
  bare `raise` bypasses that path and kills the run.

## LangGraph persistence

**`BaseCheckpointSaver`** must implement `get_tuple`, `list`, `put`,
`put_writes`, `delete_thread` and their async twins. `get`/`aget` are free.

- `list` must yield **newest first**, descending by `checkpoint_id`. This is a
  tested conformance requirement, not a convention.
- `delete_thread` is in the required base capability set, not optional.
- Conformance detects a capability by comparing the subclass method against the
  base method, so a method overridden only to `raise NotImplementedError` is
  detected as supported and then fails. Use `skip_capabilities=` instead.
- `CheckpointTuple` is `(config, checkpoint, metadata, parent_config,
  pending_writes)`. The `config["configurable"]` keys are `thread_id`,
  `checkpoint_ns` (`""` at root, `node:uuid` for subgraphs, nested joined by
  `|`), and `checkpoint_id`.
- Persist the **type tag** from `serde.dumps_typed()`, not just the bytes.
  Dropping it makes stored checkpoints unreadable.

**`BaseStore`** has only two abstract methods, `batch` and `abatch`; everything
else is sugar over them. Results **must be positionally aligned with the input
operations** — a reordered or filtered result list silently corrupts caller
state. `delete` is expressed as a `PutOp` with `value=None`. Without an
`IndexConfig`, `index=` arguments to `put` are ignored.

**LangGraph's `BaseCache`** is a different class from
`langchain_core.caches.BaseCache`. All six methods are abstract, they are batch
oriented over `(namespace, key)` tuples, and TTL is the second element of the
`set` value tuple, in seconds.

## Deep Agents backends

`BackendProtocol` covers `ls`, `read`, `grep`, `glob`, `write`, `edit`,
`delete`, `upload_files`, `download_files` plus async twins.
`SandboxBackendProtocol` adds exactly two members: an `id` property and
`execute`. Sync defaults raise `NotImplementedError`; async defaults delegate
to the sync method via `asyncio.to_thread`.

- **Expected failures go in the result's `error` field — do not raise.** Every
  result dataclass carries `error: str | None`, where `None` means success.
  Raising for a missing file or a permission error breaks the agent loop.
- `upload_files` / `download_files` must return results **in input order** and
  use the four `FileOperationError` literals: `file_not_found`,
  `permission_denied`, `is_directory`, `invalid_path`.
- `ReadResult.__post_init__` raises `ValueError` on inconsistent pagination:
  `start_line`/`end_line` are 1-indexed and set together, `1 <= start <= end`,
  `total_lines >= end_line`, and **`next_offset` must equal `end_line`**.
- `grep` takes a **literal** string, not a regex.
- `read` returns raw content; line-number gutters are added downstream, so
  adding them in the backend double-formats the output.
- Detect command failure with `ExecuteArtifact["exit_code"]`, not
  `ToolMessage.status`.
- `ExecuteResponse.output` combines stdout and stderr.
- Deep Agents is pre-1.0 and changed breakingly at 0.7: `ls_info`, `glob_info`,
  `grep_raw`, and `files_update` are gone; backends are concrete instances
  rather than factories; `virtual_mode` defaults to `True`; and `write_file` is
  create-or-replace. Treat a widened version range here with suspicion.
