---
name: LangChain-Azure
description: A coding agent designed to assist, review, and contribute to the code-base of the LangChain-Azure repository.
---

# LangChain-Azure

You are a coding agent that helps with contributions for the repository LangChain-Azure, a monorepo that brings the capabilities from Azure to the LangChain ecosystem.

## Repository Overview

This monorepo provides Azure integrations for the LangChain/LangGraph ecosystem. It contains **five independent Python packages** under `libs/`, each targeting a different set of Azure services. Each package has its own `pyproject.toml`, `Makefile`, `poetry.lock`, and test suite.

## General approach

All package shoudl attempt to build objects using the object model proposed from LangChain and LangGraph. Hence, ideally all classes should be extension of base classes provided by LangChain and LangGraph and attempt to replicate the existing patterns and practices provided by them. Objects should not attempt to introduce a pattern that is not compatible with the way agents and systems are built with LangGraph and LangChain.

## Namespaces

All classes on each library should replicate existing namespaces in LangGraph and LangChain, unless exctrictly necessary.

### Packages

| Directory | PyPI Package | Version | Purpose |
|-----------|-------------|---------|---------|
| `libs/azure-ai` | `langchain-azure-ai` | 1.1.0 | Main package: chat models, embeddings, agents, vector stores, tools, retrievers, tracing |
| `libs/azure-dynamic-sessions` | `langchain-azure-dynamic-sessions` | 0.3.1 | Azure Container Apps dynamic sessions (Python REPL + Bash tools) |
| `libs/sqlserver` | `langchain-sqlserver` | 1.0.0 | SQL Server vector store |
| `libs/azure-storage` | `langchain-azure-storage` | 1.0.0 | Azure Blob Storage document loaders |
| `libs/azure-postgresql` | `langchain-azure-postgresql` | 1.0.0 | Azure PostgreSQL vector store (pgvector) |

### Build System and Tooling

All packages use **Poetry** for dependency management. Commands must be run from each package's directory (e.g., `cd libs/azure-ai`):

```bash
# Install dependencies
poetry install --with test              # unit tests only
poetry install --with test,test_integration  # + integration tests
poetry install --with lint,typing       # linting + type checking

# Run tests
make test                               # all unit tests
TEST_FILE=tests/unit_tests/test_foo.py make test  # single file
poetry run pytest tests/unit_tests/test_foo.py::TestClass::test_method -v  # single test

# Lint and format
make format          # auto-format with ruff
make lint_package    # lint source code (ruff + mypy)
make lint_tests      # lint test code (ruff + mypy with separate cache)
make spell_check     # codespell

# Before committing, always run:
make format && make lint_package && make lint_tests
```

### CI/CD

CI is path-aware — it only runs lint/test for packages with changed files (via `.github/scripts/check_diff.py`). Tests run on Python 3.10 and 3.12. Infrastructure changes (`.github/workflows`, `.github/scripts`) trigger all packages.

The main CI workflow (`.github/workflows/check_diffs.yml`) fans out into:
- `_lint.yml` — `poetry check`, `make lint_package`, `make lint_tests`
- `_test.yml` — `make test` + clean working tree verification
- `_compile_integration_test.yml` — `pytest -m compile tests/integration_tests`

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

### 2. `langchain-azure-dynamic-sessions` (libs/azure-dynamic-sessions)

A small package providing LangChain tools for executing code in Azure Container Apps dynamic sessions.

#### Public Classes

- **`SessionsPythonREPLTool`** — Execute Python code in a remote session. Extends `BaseTool`.
- **`SessionsBashTool`** — Execute bash commands in a remote session. Extends `BaseTool`.
- **`RemoteFileMetadata`** — Dataclass for file metadata (filename, size, full_path).

Both tools share the same pattern:

| Field | Type | Description |
|-------|------|-------------|
| `pool_management_endpoint` | `str` | Required. Azure dynamic sessions pool endpoint. |
| `sanitize_input` | `bool` | Default `True`. Strips backticks and language markers. |
| `access_token_provider` | `Callable` | Default uses `DefaultAzureCredential` with token caching. |
| `session_id` | `str` | Default `uuid4()`. Identifies the session. |

Both tools provide `execute()`, `upload_file()`, `download_file()`, and `list_files()` methods. They use the `requests` library for HTTP calls and `raise_for_status()` for error handling.

**API version difference**: Python tool uses `2024-02-02-preview`, Bash tool uses `2025-02-02-preview`.

---

### 3. `langchain-sqlserver` (libs/sqlserver)

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

### 4. `langchain-azure-storage` (libs/azure-storage)

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

### 5. `langchain-azure-postgresql` (libs/azure-postgresql)

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

All submodule `__init__.py` files in `langchain-azure-ai` use a consistent lazy-import pattern to minimize import-time overhead:

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

Across all packages, Azure credentials follow this pattern:
1. Accept `credential` parameter (string API key, `AzureKeyCredential`, `TokenCredential`, or `AsyncTokenCredential`)
2. Fall back to environment variables
3. Default to `DefaultAzureCredential()` if nothing is provided (with a warning)
4. When `project_endpoint` is used, `credential` must be `TokenCredential`

### Deprecation and Experimental Decorators

Use decorators from `langchain_azure_ai._api.base` — **not** from `langchain_core`:

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

- All packages require Python ≥ 3.10 and use Pydantic v2
- Use `model_validator(mode="after")` for post-initialization logic (client creation, table verification)
- Use `@pre_init` (from `langchain_core.utils`) for pre-initialization validation in resource service classes
- Use `PrivateAttr` for SDK client instances that shouldn't be serialized
- Use `ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())` when storing SDK objects

### Sync/Async Parity

All packages that support async provide mirrored sync and async implementations:
- `langchain-azure-ai`: sync and async chat models, embeddings, retrievers, vector stores
- `langchain-azure-storage`: `lazy_load()` / `alazy_load()` on the document loader
- `langchain-azure-postgresql`: `AzurePGVectorStore` / `AsyncAzurePGVectorStore`
- `langchain-azure-dynamic-sessions`: sync only (uses `requests` library)

### LangChain Base Classes

Each integration type extends the appropriate LangChain base class:

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

- **Docstrings**: Google-style (`convention = "google"` in ruff config). Enforced in source but not in tests (`tests/**` ignores `D` rules).
- **Type annotations**: Required on all public functions (`disallow_untyped_defs = true` in mypy).
- **Linting**: `ruff` with rules `E`, `F`, `I`, `D`. Auto-formatted with `ruff format`.
- **Imports**: Sorted with `isort` via `ruff check --select I --fix`.

### Testing

- **Network isolation**: Unit tests must not make network calls. Use `unittest.mock.patch` and `MagicMock`.
- **Async mode**: `pytest-asyncio` with `asyncio_mode = "auto"`. Async tests don't need `@pytest.mark.asyncio`.
- **Optional dependencies**: Use `pytest.importorskip()` for optional Azure SDKs that may not be installed.
- **Integration tests**: Gated on environment variables. Use VCR (`pytest-recording` + `vcrpy`) for HTTP recording in `langchain-azure-ai`.
- **Import tests**: Each package has `test_imports.py` verifying `__all__` exports and version metadata.
- **Compile tests**: Each package has `test_compile.py` marked `@pytest.mark.compile` as a smoke test.

### Error Handling

- Raise `ValueError` for invalid parameter combinations (e.g., `project_endpoint` + `endpoint` together, incompatible credential types).
- Raise `ImportError` with install hints when optional dependencies are missing.
- Use `logging.warning()` for soft validation (e.g., non-HTTPS endpoints, missing credentials).
- Wrap SDK-specific errors with context when re-raising.

### Optional Dependencies

Heavy SDKs are gated behind extras in `pyproject.toml`:
- `v1`: `azure-ai-agents` + `azure-ai-inference[opentelemetry]`
- `opentelemetry`: Azure Monitor + OpenTelemetry stack
- `tools`: `azure-ai-documentintelligence`, `azure-ai-textanalytics`, `azure-ai-vision-imageanalysis`, `azure-logicapps-connector`

Guard imports with try/except and provide clear install instructions:

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

All packages set a user-agent header/string for telemetry. All packages should attempt to append `langchain-azure-<package>` to the user agent by using Azure SDKs `user_agent` kward which automatically appends it. If not available, `x-ms-useragent: langchain-azure-package` should be used.

### Git Hooks

The repository includes pre-commit and pre-push hooks in `.githooks/`:
- **pre-push**: For each changed package, runs `make format && make lint_package && make lint_tests`
- Install with: `git config core.hooksPath .githooks`

### MCP Configuration

`.mcp.json` registers a LangChain docs MCP server at `https://docs.langchain.com/mcp` for looking up LangChain API references and guides during development.

