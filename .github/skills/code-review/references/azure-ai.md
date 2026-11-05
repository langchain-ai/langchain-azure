# `langchain-azure-ai` (`libs/azure-ai`)

The largest package: chat models, embeddings, Agent Service, LangGraph agent
hosting, content-safety middleware, Azure AI Search vector stores, tools,
document loaders, retrievers, evaluation, and tracing. It has its own
`AGENTS.md` — read it, and treat its public-API and docstring rules as binding.

## Public surface

Submodule `__init__.py` files use a lazy-import pattern. A new public symbol
must appear in **all three** of the `TYPE_CHECKING` import, `__all__`, and
`_module_lookup`. Missing one produces a symbol that fails at attribute access
or is absent from `__all__`, and the import tests catch only part of that.

Every exported name is a compatibility contract: import path, signature,
defaults, return type, raised exceptions. Optional extras (`tools`,
`opentelemetry`, `hosting`, `v1`) must not become required — importing the base
package with none of them installed must still work, and the guard must raise
`ImportError` naming the extra to install.

## Chat models and embeddings

Apply the `BaseChatModel` contract in
[ecosystem-contracts.md](ecosystem-contracts.md), and pay particular attention
to two things this package gets wrong easily:

- Classes keyed on `azure_deployment`, `deployment_name`, or `model_id` **must**
  override `_get_ls_params`, because the base resolves only `model` /
  `model_name` and will otherwise report the wrong model to tracing.
- OpenAI-compatible classes inherit from `ChatOpenAI` / `OpenAIEmbeddings`.
  When a change overrides a method there, check it against the parent's
  behavior, not against `BaseChatModel` alone.

## Agent Service and hosting

- Agent nodes run remotely but compose into ordinary graphs. Check thread and
  conversation identity, state propagation, streaming and event ordering,
  interrupt / human-in-the-loop handling, and cleanup of remote resources.
- The hosting layer implements a protocol boundary: request-to-state and
  state-to-event conversion, durable checkpoint references, task and
  conversation isolation, replay and resume, disconnect and cancellation, and
  exactly-once terminal events. Credentials and private state must never reach
  a response event.
- Hosting maintains process-global state (user-agent prefixes, feature flags,
  `AZURE_HTTP_USER_AGENT`). Changes there must stay idempotent under repeated
  import and must not clobber a value the application set itself.
- The `_foundry_checkpoint_saver` is a LangGraph checkpointer: the
  `BaseCheckpointSaver` rules in
  [ecosystem-contracts.md](ecosystem-contracts.md) apply in full, including
  descending `list` order and serializer type tags.

## Middleware and content safety

Middleware must preserve message ordering and types and must not silently
bypass a configured safety check. Review whether a failure fails open or
closed, and say which the change chooses — a content-safety check that fails
open on a transient service error is a High-severity finding unless that is the
documented, intended behavior.

## Azure AI Search

Preserve index schema compatibility, vector dimensions, field mappings, filter
syntax, scoring configuration, pagination, and metadata round-tripping. The
`VectorStore` rules in [ecosystem-contracts.md](ecosystem-contracts.md) apply,
especially relevance-score direction and the `add_texts` / `add_documents`
delegation trap.

## Versioned and deprecated APIs

`agents/v1` and `agents/_v1` exist for compatibility with Microsoft Foundry
classic and are gated behind the `v1` extra. New functionality belongs in the
current API; expanding the deprecated surface is a design finding. Deprecations
in this package use the local `langchain_azure_ai._api.base` decorators
(`deprecated`, `experimental`), not `langchain_core._api`.

## Tests

- Unit tests are network-isolated with `pytest-socket`; a new unit test that
  reaches a service will fail in CI.
- Optional SDKs are stubbed in `sys.modules` at the **top of the test file**,
  before importing the module under test, because the tool module raises on a
  missing import at import time.
- When a tool imports an SDK at module level, patching that name only during
  construction does not keep it patched for later method calls. The call must
  happen inside the `with patch(...)` block.
- `FDPResourceService.validate_environment` reads `AZURE_AI_PROJECT_ENDPOINT`.
  Tests that construct a service without an explicit `endpoint` will fail when
  that variable is set in the environment. Passing `endpoint` explicitly or
  deleting the variable is the fix, so do not flag those as redundant.
