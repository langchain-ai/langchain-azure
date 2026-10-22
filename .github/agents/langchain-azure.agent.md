---
name: LangChain-Azure
description: A coding agent designed to assist, review, and contribute to the code-base of the LangChain-Azure repository.
---

# LangChain-Azure

You are a coding agent that helps with contributions for the repository LangChain-Azure, a monorepo that brings the capabilities from Azure to the LangChain ecosystem.

## Repository Overview

This monorepo provides Azure integrations for the LangChain/LangGraph ecosystem. It contains **seven independently maintained Python packages** under `libs/`, each with its own `pyproject.toml`, `Makefile`, `uv.lock`, and test suite. Follow the root `AGENTS.md` and any applicable package-local instructions before changing a package; consult that package's `pyproject.toml` for its current version and dependencies.

## General approach

Packages should follow the object models and supported extension points of LangChain, LangGraph, or Deep Agents as appropriate. Extend their base classes where the integration contract calls for it, rather than introducing incompatible agent or system patterns.

## Namespaces

Keep public namespaces consistent with the relevant upstream ecosystem and with the package's existing import paths.

### Packages

| Directory | PyPI Package | Purpose |
|-----------|-------------|---------|
| `libs/azure-ai` | `langchain-azure-ai` | Microsoft Foundry models, agents, AI services tools, Azure AI Search, and tracing |
| `libs/azure-compute` | `langchain-azure-compute` | Azure Container Apps dynamic-sessions tools and Deep Agents backends for dynamic sessions and sandboxes |
| `libs/azure-cosmosdb` | `langchain-azure-cosmosdb` | Cosmos DB / Azure DocumentDB vector search, semantic cache, chat history, and LangGraph persistence |
| `libs/azure-dynamic-sessions` | `langchain-azure-dynamic-sessions` | **Deprecated** legacy dynamic-sessions integrations; use `langchain-azure-compute` for new work |
| `libs/sqlserver` | `langchain-sqlserver` | SQL Server vector store |
| `libs/azure-storage` | `langchain-azure-storage` | Azure Blob Storage document loaders |
| `libs/azure-postgresql` | `langchain-azure-postgresql` | Azure PostgreSQL vector store (pgvector) |

### Build System and Tooling

All packages have **uv** lockfiles. Run commands from the affected package directory (e.g., `cd libs/azure-ai`), and check its `Makefile` and local instructions for supported targets. Most packages use the following workflow:

```bash
# Install dependencies
uv sync --group test                          # unit tests only
uv sync --group test --group test_integration # + integration tests
uv sync --group lint --group typing           # linting + type checking
uv sync --all-extras --all-groups             # everything

# Run tests (most Makefile targets wrap `uv run --frozen`)
make test                               # all unit tests
TEST_FILE=tests/unit_tests/test_foo.py make test  # single file
uv run --frozen --all-extras --group test pytest tests/unit_tests/test_foo.py::TestClass::test_method -v  # single test

# Lint and format
make format          # auto-format with ruff
make lint_package    # lint source code (ruff + mypy)
make lint_tests      # lint test code (ruff + mypy with separate cache)
make spell_check     # codespell

# Keep the lockfile authoritative
uv lock --check      # verify uv.lock is up to date (run in CI)
uv lock              # regenerate uv.lock after changing dependencies
```

`libs/azure-postgresql` uses `uv run tox -e py`, `uv run tox -m lint`, and `uv run tox -m type`; its Makefile targets delegate to tox and are not equivalent to the other packages' lint targets. Use the package-specific commands when working there.

### CI/CD

CI is path-aware — it runs lint/test for packages with changed files (via `.github/scripts/check_diff.py`, which lists all seven packages). Tests run on Python 3.11 and 3.14. Relevant infrastructure changes trigger all packages.

The main CI workflow (`.github/workflows/check_diffs.yml`) fans out into:
- `_lint.yml` — `uv lock --check`, `make lint_package`, `make lint_tests`
- `_test.yml` — `make test` + clean working tree verification
- `_compile_integration_test.yml` — compile smoke tests where a package provides them

Release uses trusted publishing via `_release.yml` with pre-release validation on Test PyPI.

---

## Package Architectures

### 1. `langchain-azure-ai` (libs/azure-ai)

The largest and most complex package. Provides integrations for Azure AI Foundry services.

#### Module Structure

```
langchain_azure_ai/
├── __init__.py              # Docstring only, no exports
├── _api/base.py             # @deprecated() and @experimental() decorators
├── _resources.py            # Base classes for service connectivity
├── agents/                  # Azure AI Foundry agent service (V1 + V2)
│   ├── __init__.py          # Default surface → V2
│   ├── v1/                  # Public re-exports for V1 (deprecated)
│   ├── v2/                  # Public re-exports for V2
│   ├── _v1/                 # Private V1 implementation
│   ├── _v2/                 # Private V2 implementation
│   └── prebuilt/            # Prebuilt agent nodes and tools
├── callbacks/tracers/       # OpenTelemetry tracing
├── chat_history/            # Chat message history (Cosmos DB, AI Memory)
├── chat_models/             # Chat completions (OpenAI-compatible, Inference SDK)
├── embeddings/              # Embeddings (OpenAI-compatible, Inference SDK)
├── query_constructors/      # Cosmos DB NoSQL query translation
├── retrievers/              # Azure AI Search, AI Memory retrievers
├── tools/                   # AI services tools + toolkit
│   └── services/            # Document Intelligence, Image Analysis, Text Analytics
├── utils/                   # Shared helpers (env, math, JSON encoding)
└── vectorstores/            # Azure AI Search, Cosmos DB (Mongo + NoSQL), caches
```

#### Resource Service Base Classes (`_resources.py`)

The package provides a hierarchy of base classes for connecting to Azure services:

```
FDPResourceService (BaseModel)
├── AIServicesService          # service = "cognitive_services"
└── ModelInferenceService      # service = "inference"
```

**`FDPResourceService`** provides the common fields used across many classes:

| Field | Type | Description |
|-------|------|-------------|
| `project_endpoint` | `Optional[str]` | Azure AI Foundry project endpoint. When set, `credential` must be `TokenCredential`. |
| `endpoint` | `Optional[str]` | Direct service endpoint URL. |
| `credential` | `Optional[str \| AzureKeyCredential \| TokenCredential]` | API key or Azure credential. Defaults to `DefaultAzureCredential()`. |
| `api_version` | `Optional[str]` | Azure API version. |
| `client_kwargs` | `Dict[str, Any]` | Additional kwargs passed to the underlying SDK client. |

The `validate_environment` pre-init validator resolves values from environment variables:
- `AZURE_AI_INFERENCE_CREDENTIAL` → `credential`
- `AZURE_AI_PROJECT_ENDPOINT` → `project_endpoint`
- `AZURE_AI_INFERENCE_ENDPOINT` → `endpoint`

When `project_endpoint` is set, the validator calls `get_service_endpoint_from_project()` to resolve the actual service endpoint. It also sets `user_agent = "langchain-azure-ai"` in `client_kwargs`.

Tools in `tools/services/` inherit from both `BaseTool` and `AIServicesService`, gaining endpoint/credential resolution automatically:

```python
class AzureAIDocumentIntelligenceTool(BaseTool, AIServicesService):
    ...
```

**Important limitation:** `AIServicesService` (and `FDPResourceService`) is only appropriate for tools whose underlying SDK accepts an **endpoint URL** for authentication. SDKs that use a **region + subscription key** model (e.g., `azure-cognitiveservices-speech`) are incompatible because `FDPResourceService.validate_environment` always requires a resolvable endpoint. For such tools, extend `BaseTool` directly, define your own `credential`, `region`, and `endpoint` fields, and implement env-var resolution via `model_validator(mode="before")` and client initialization via `model_validator(mode="after")`.

However, if the underlying SDK also accepts an **endpoint URL** in place of a region (which most Azure Cognitive Services SDKs do, with the endpoint following the pattern `https://<region>.api.cognitive.microsoft.com/`), it is preferable to **require `endpoint` instead of `region`** and inherit `AIServicesService` normally. This keeps the tool consistent with the rest of the package and automatically enables the `project_endpoint` pattern for free. Only fall back to extending `BaseTool` directly when the SDK truly has no endpoint-based auth path.

#### OpenAI-compatible Classes (Chat Models + Embeddings)

`AzureAIOpenAIApiChatModel` and `AzureAIOpenAIApiEmbeddingsModel` extend `langchain_openai` classes and use a separate credential resolution function `_configure_openai_credential_values()` that supports:

- **Project-endpoint pattern** (recommended): Uses `AIProjectClient` to obtain pre-configured OpenAI clients. Requires `TokenCredential`.
- **Direct endpoint pattern**: Maps credential to `api_key` or `azure_ad_token_provider`.

Environment variable resolution priority (highest to lowest):
1. Constructor parameters (`project_endpoint`, `endpoint`, `model`, `credential`, `api_version`)
2. `AZURE_AI_PROJECT_ENDPOINT`
3. `AZURE_AI_OPENAI_ENDPOINT`
4. `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_VERSION` / `AZURE_OPENAI_DEPLOYMENT_NAME`

Providing both `project_endpoint` and `endpoint` as constructor parameters raises `ValueError`.

#### Agent Service Versioning (V1 / V2)

The agents module has two parallel implementations:

- **V1** (`agents/_v1/`): Uses `azure-ai-agents` SDK with threads/runs pattern. **Deprecated.**
- **V2** (`agents/_v2/`): Uses `azure-ai-projects >= 2.0` with Responses/Conversations API (OpenAI SDK types). **Current.**

The default import path (`from langchain_azure_ai.agents import AgentServiceFactory`) resolves to **V2**. V1 requires explicit import from `langchain_azure_ai.agents.v1`.

Implementation lives in private `_v1/` and `_v2/` directories. Public API directories (`v1/`, `v2/`, `prebuilt/`) only contain `__init__.py` files that re-export via lazy imports.

Key V2 classes:
- `AgentServiceFactory` — factory for creating LangGraph agent nodes
- `PromptBasedAgentNode` — the agent node that proxies to the Azure AI Foundry agent service
- `AgentServiceAgentState` — the LangGraph state schema
- `AgentServiceBaseTool`, `ImageGenTool`, `CodeInterpreterTool`, `MCPTool` — tool wrappers

V2 supports middleware (`AgentMiddleware` with `before_agent`, `after_agent`, `wrap_tool_call` hooks) and MCP approval flows via `interrupt()`.

#### Tracing (`callbacks/tracers/inference_tracing.py`)

`AzureAIOpenTelemetryTracer` is a comprehensive LangChain callback handler that produces OpenTelemetry spans for LLM operations, agents, tools, and retrievers. Supports Azure Monitor auto-configuration, content redaction, and span parenting across LangGraph nodes.

#### Deprecated Classes

The following classes use the legacy `azure-ai-inference` SDK and are deprecated:
- `AzureAIChatCompletionsModel` (use `AzureAIOpenAIApiChatModel`)
- `AzureAIEmbeddingsModel` (use `AzureAIOpenAIApiEmbeddingsModel`)
- V1 agents classes (use V2)

---

### 2. `langchain-azure-compute` (libs/azure-compute)

Integrations for two distinct Azure Container Apps products: ephemeral dynamic sessions (`Microsoft.App/sessionPools`) and stateful sandboxes (`Microsoft.App/sandboxGroups`). Feature-specific dependencies are provided by the `dynamic-sessions` and `sandboxes` extras; both modules ship in the wheel, and each guards its optional imports with an install hint.

- `langchain_azure_compute.dynamic_sessions` exports `SessionsPythonREPLTool` and `SessionsBashTool` for remote code execution using a pool management endpoint and token provider.
- `langchain_azure_compute.dynamic_sessions.backends` exports `SessionsBashBackend`, a Deep Agents `SandboxBackendProtocol` implementation for Shell-typed session pools.
- `langchain_azure_compute.sandboxes` exports `ACASandbox`, a Deep Agents backend wrapping a caller-supplied Azure Container Apps `SandboxClient`. The caller owns the client and sandbox lifecycle. The Deep Agents backends are beta.

Use `pip install "langchain-azure-compute[dynamic-sessions]"` or `pip install "langchain-azure-compute[sandboxes]"` as appropriate. Do not conflate the products' endpoints, resource lifecycles, or data planes. See the [package README](../../libs/azure-compute/README.md) for file-operation and output-size limits.

---

### 3. `langchain-azure-cosmosdb` (libs/azure-cosmosdb)

Azure Cosmos DB integrations for both LangChain and LangGraph. The package uses a `src/langchain_azure_cosmosdb/` layout with public exports in `__init__.py` and async implementations under `aio/`.

- **LangChain:** `AzureCosmosDBNoSqlVectorSearch` and `AsyncAzureCosmosDBNoSqlVectorSearch` support vector, full-text, and hybrid search; `AzureDocumentDBVectorSearch` and `AzureCosmosDBMongoVCoreVectorSearch` cover MongoDB-compatible stores. The package also provides sync/async semantic caches and chat message history, plus a NoSQL query translator.
- **LangGraph:** `CosmosDBSaverSync` / `CosmosDBSaver` provide checkpointing, `CosmosDBCacheSync` / `CosmosDBCache` provide graph caching, and `CosmosDBStore` / `AsyncCosmosDBStore` provide long-term memory.

Use the appropriate sync or async client and import from `langchain_azure_cosmosdb`; its exports are not the lazy-import pattern used in `langchain-azure-ai`. Consult the [package README](../../libs/azure-cosmosdb/README.md) for index policies, credentials, and setup requirements. Prefer this package for new Cosmos DB integrations; preserve existing `langchain-azure-ai` import paths when maintaining their consumers.

---

### 4. `langchain-azure-dynamic-sessions` (libs/azure-dynamic-sessions)

**Deprecated; no further fixes planned.** Dynamic-sessions tools and `SessionsBashBackend` are now maintained in `langchain-azure-compute`. Existing imports remain in this package, but use `langchain_azure_compute.dynamic_sessions` for new tools and `langchain_azure_compute.dynamic_sessions.backends` for the backend. The replacement also supports Deep Agents versions for which this legacy backend is broken. See the [migration guide](../../libs/azure-dynamic-sessions/README.md) before changing consumers.

---

### 5. `langchain-sqlserver` (libs/sqlserver)

SQL Server vector store using the `VECTOR` data type and `VECTOR_DISTANCE()` function.

#### Public Classes

- **`SQLServer_VectorStore`** — Main vector store. Extends `langchain_core.vectorstores.VectorStore`.

| Field | Type | Description |
|-------|------|-------------|
| `connection_string` | `str` | ODBC connection string or SQLAlchemy URL |
| `embedding_function` | `Embeddings` | LangChain embeddings instance |
| `embedding_length` | `int` | Embedding vector dimension |
| `table_name` | `str` | Default `"sqlserver_vectorstore"` |
| `db_schema` | `Optional[str]` | Database schema |
| `distance_strategy` | `DistanceStrategy` | Default `COSINE`. Also supports `EUCLIDEAN`, `DOT`. |
| `batch_size` | `int` | Default `100`. Insert batch size. |

Uses SQLAlchemy with `pyodbc`. Supports Entra ID authentication (auto-detected from connection string), username/password, and trusted connection. Custom `VectorType` SQLAlchemy type maps to SQL Server's `vector(n)`.

Filtering supports `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$like`, `$between`, `$and`, `$or` operators on JSON metadata.

---

### 6. `langchain-azure-storage` (libs/azure-storage)

Azure Blob Storage document loader.

#### Public Classes

- **`AzureBlobStorageLoader`** — Load documents from Azure Blob Storage. Extends `BaseLoader`. Decorated with `@beta()`.

| Field | Type | Description |
|-------|------|-------------|
| `account_url` | `str` | Azure Blob Storage account URL |
| `container_name` | `str` | Container to read from |
| `blob_names` | `Optional[str \| Iterable[str]]` | Explicit blob names (mutually exclusive with `prefix`) |
| `prefix` | `Optional[str]` | Prefix filter for listing blobs |
| `credential` | SDK credential type | Azure credential. Defaults to `DefaultAzureCredential()`. |
| `loader_factory` | `Optional[Callable]` | Custom loader factory for blob parsing |

Supports both sync (`lazy_load()`) and async (`alazy_load()`) loading. Default behavior decodes blob content as UTF-8. Custom `loader_factory` writes blobs to temp files and delegates parsing. Automatically filters out ADLS directory markers.

---

### 7. `langchain-azure-postgresql` (libs/azure-postgresql)

PostgreSQL vector store using `pgvector` with Azure-specific connection pooling and authentication.

#### Public Classes

**Connection layer:**
- `ConnectionInfo` / `AsyncConnectionInfo` — Pydantic models with credential + connection details
- `AzurePGConnectionPool` / `AsyncAzurePGConnectionPool` — Connection pools with Azure token refresh

**Vector store:**
- `AzurePGVectorStore` — Sync vector store. Extends `BaseModel` + `VectorStore`.
- `AsyncAzurePGVectorStore` — Async vector store. Same fields, async operations.

| Field | Type | Description |
|-------|------|-------------|
| `embedding` | `Embeddings` | LangChain embeddings instance |
| `connection` | pool or connection | Direct connection or connection pool |
| `table_name` | `str` | Default `"langchain"` |
| `schema_name` | `str` | Default `"public"` |
| `embedding_column` | `str` | Default `"embedding"` |
| `embedding_type` | `VectorType` | Default `vector`. Also supports `halfvec`, `bit`. |
| `embedding_dimension` | `int` | Default `1536` |
| `embedding_index` | `Algorithm` | Default `DiskANN(vector_cosine_ops)` |
| `metadata_columns` | `str \| list` | Default `"metadata"` (JSONB). Can be explicit column list. |

**Index algorithms:** `DiskANN`, `HNSW`, `IVFFlat` — each with typed search params and build settings. All extend `Algorithm[SP]` (generic over search params type).

Uses `psycopg` with safe SQL composition (`psycopg.sql.Identifier`, `Literal`, `Placeholder`). Sync and async implementations are mirrored closely. Supports product quantization reranking.

---

## Cross-Package Patterns

### Lazy Import Pattern

Submodule `__init__.py` files in `langchain-azure-ai` use a lazy-import pattern to minimize import-time overhead:

```python
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.module._private import NewClass

__all__ = ["NewClass"]

_module_lookup = {
    "NewClass": "langchain_azure_ai.module._private",
}

def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

When adding new public symbols, add to all three: `TYPE_CHECKING` import, `__all__`, and `_module_lookup`.

### Credential Resolution

Authentication is package- and service-specific. `langchain-azure-ai` resource services resolve a credential and endpoint as described above; their `project_endpoint` requires a `TokenCredential`. `langchain-azure-compute` dynamic sessions use a pool endpoint and token provider, while sandboxes receive an already constructed SDK client. Cosmos DB integrations accept service-specific clients, endpoints, and credentials; SQL Server and PostgreSQL use database connection configurations. Follow the selected package's API rather than applying the Foundry credential rules across all packages.

### Deprecation and Experimental Decorators

In `langchain-azure-ai`, use decorators from `langchain_azure_ai._api.base` — **not** from `langchain_core`:

```python
from langchain_azure_ai._api.base import deprecated, experimental

@deprecated("0.2.0", alternative="NewClass", removal="1.0.0")
class OldClass:
    pass

@experimental()
class PreviewClass:
    pass
```

- `@deprecated()` emits `DeprecationWarning` with `langchain-azure-ai=={version}` messaging
- `@experimental()` emits `ExperimentalWarning` with Azure preview terms link
- Both set `__deprecated__`/`__experimental__` attributes for introspection
- Helper functions: `is_deprecated()`, `is_experimental()`, `get_deprecation_message()`, `get_experimental_message()`
- Warning control: `suppress_deprecation_warnings()`, `surface_deprecation_warnings()`, etc.

### Pydantic Usage

- All packages support Python 3.11 through 3.14. For Pydantic models, use the v2 API.
- Use `model_validator(mode="after")` for post-initialization logic in Pydantic models (client creation, table verification)
- In `langchain-azure-ai` resource service classes, use `@pre_init` (from `langchain_core.utils`) for pre-initialization validation
- Use `PrivateAttr` for SDK client instances that shouldn't be serialized
- Use `ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())` when storing SDK objects

### Sync/Async Parity

Preserve each package's supported sync/async contract:
- `langchain-azure-ai`: sync and async chat models, embeddings, retrievers, vector stores
- `langchain-azure-compute`: dynamic-sessions tools use synchronous requests; the Deep Agents backends also expose async operations
- `langchain-azure-cosmosdb`: sync and async NoSQL vector stores, semantic caches, chat history, and LangGraph checkpoint/cache/store APIs
- `langchain-azure-storage`: `lazy_load()` / `alazy_load()` on the document loader
- `langchain-azure-postgresql`: `AzurePGVectorStore` / `AsyncAzurePGVectorStore`
- `langchain-azure-dynamic-sessions`: legacy synchronous tools; direct new feature work to `langchain-azure-compute`

### LangChain Base Classes

LangChain integrations extend the appropriate LangChain base class; LangGraph checkpointers, caches, and stores and Deep Agents backends follow their own upstream contracts:

| Integration | Base Class |
|------------|------------|
| Chat models | `BaseChatModel` or `ChatOpenAI` |
| Embeddings | `Embeddings` or `OpenAIEmbeddings` |
| Vector stores | `VectorStore` |
| Document loaders | `BaseLoader` |
| Tools | `BaseTool` |
| Toolkits | `BaseToolkit` |
| Retrievers | `BaseRetriever` |
| Chat history | `BaseChatMessageHistory` |
| Caches | `BaseCache` |

---

## Coding Standards and Best Practices

### Code Style

- **Docstrings**: Follow the affected package's conventions. Most configure Google-style docstrings in ruff; PostgreSQL's local instructions specify Sphinx-style.
- **Type annotations**: Required on all public functions (`disallow_untyped_defs = true` in mypy).
- **Linting**: `ruff` with rules `E`, `F`, `I`, `D`. Auto-formatted with `ruff format`.
- **Imports**: Sorted with `isort` via `ruff check --select I --fix`.

### Testing

- **Network isolation**: Unit tests must not make network calls. Use `unittest.mock.patch` and `MagicMock`.
- **Async mode**: `pytest-asyncio` with `asyncio_mode = "auto"`. Async tests don't need `@pytest.mark.asyncio`.
- **Optional dependencies**: Use `pytest.importorskip()` for optional Azure SDKs that may not be installed.
- **Integration tests**: Gated on environment variables. Use VCR (`pytest-recording` + `vcrpy`) for HTTP recording in `langchain-azure-ai`.
- **Import tests**: Where present, `test_imports.py` checks public exports and version metadata; use the package's actual test layout.
- **Compile tests**: Where present, `test_compile.py` provides a `@pytest.mark.compile` smoke test.
- **Stubbing optional SDK dependencies**: When an optional SDK (e.g., `azure-cognitiveservices-speech`) is not installed in the test environment, register a minimal stub module in `sys.modules` at the **top of the test file**, before any import of the tool module. This prevents the top-level `ImportError` guard in the tool from triggering during test collection.
- **Mock scope for module-level imports**: When a tool imports an SDK at module level (e.g., `import azure.cognitiveservices.speech as speechsdk`), patching that name during construction does **not** keep it active during later method calls. Any test that calls a method relying on the patched module must wrap that call inside the `with patch(...):` block, not just the constructor.
- **`AZURE_AI_PROJECT_ENDPOINT` in the developer environment**: `FDPResourceService.validate_environment` reads `AZURE_AI_PROJECT_ENDPOINT` from the environment and, when set, requires `credential` to be a `TokenCredential`. If this env var is present in the developer's shell, unit tests that instantiate any `AIServicesService` subclass **without an explicit `endpoint`** will fail with a `ValidationError` (because `endpoint=None` causes the validator to fall through to the project-endpoint path). The fix is either: (a) always pass `endpoint` explicitly in tests — when `endpoint` is set, the project-endpoint path is never triggered; or (b) use `monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)` in tests that rely on env-var resolution of the endpoint. When running tests interactively without `make test`, use `uv run --frozen --all-extras --group test pytest -o asyncio_mode=auto -o "addopts="` to bypass the `--strict-config` flag that blocks the `asyncio_mode` warning.

### Error Handling

- Raise `ValueError` for invalid parameter combinations (e.g., `project_endpoint` + `endpoint` together, incompatible credential types).
- Raise `ImportError` with install hints when optional dependencies are missing.
- Use `logging.warning()` for soft validation (e.g., non-HTTPS endpoints, missing credentials).
- Wrap SDK-specific errors with context when re-raising.

### Optional Dependencies

In `langchain-azure-ai`, heavy SDKs are gated behind extras in `pyproject.toml`:
- `v1`: `azure-ai-agents` + `azure-ai-inference[opentelemetry]`
- `opentelemetry`: Azure Monitor + OpenTelemetry stack
- `tools`: `azure-ai-documentintelligence`, `azure-ai-textanalytics`, `azure-ai-vision-imageanalysis`, `azure-logicapps-connector`

Guard optional imports and provide clear install instructions. Follow the affected package's pattern (for example, `langchain-azure-compute` checks dependencies with `find_spec` rather than catching unrelated import failures):

```python
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
except ImportError as ex:
    raise ImportError(
        "To use Azure AI Document Intelligence tool, please install the "
        "'azure-ai-documentintelligence' package: "
        "`pip install azure-ai-documentintelligence` or install the 'tools' "
        "extra: `pip install langchain-azure-ai[tools]`"
    ) from ex
```

### User-Agent Tracking

When creating Azure SDK clients or service requests, follow the affected package's user-agent convention (for example, SDK `user_agent` in Cosmos DB or a `User-Agent` header in compute's dynamic sessions). Do not override a caller-supplied client's configuration.

### Git Hooks

The repository includes pre-commit and pre-push hooks in `.githooks/`:
- **pre-push**: For changed packages whose Makefiles provide all three targets, runs `make format && make lint_package && make lint_tests`; otherwise skips that package
- Install with: `git config core.hooksPath .githooks`

### MCP Configuration

`.mcp.json` registers a LangChain docs MCP server at `https://docs.langchain.com/mcp` for looking up LangChain API references and guides during development.
