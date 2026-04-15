# Copilot Instructions for langchain-azure

## Repository Overview

Monorepo providing Azure integrations for LangChain/LangGraph. Contains five independent Python packages under `libs/`:

| Directory | PyPI Package | Purpose |
|-----------|-------------|---------|
| `libs/azure-ai` | `langchain-azure-ai` | Main package: chat models, embeddings, agents, vector stores, tools |
| `libs/azure-dynamic-sessions` | `langchain-azure-dynamic-sessions` | Azure Dynamic Sessions integration |
| `libs/sqlserver` | `langchain-sqlserver` | SQL Server vector store |
| `libs/azure-storage` | `langchain-azure-storage` | Azure Storage integration |
| `libs/azure-postgresql` | `langchain-azure-postgresql` | Azure PostgreSQL integration |

Each package has its own `pyproject.toml`, `Makefile`, `poetry.lock`, and test suite. Always `cd` into the specific package directory before running commands.

## Build, Test, and Lint

All packages use **Poetry** for dependency management. Commands run from each package's directory (e.g., `cd libs/azure-ai`):

```bash
# Install dependencies
poetry install --with test              # unit tests
poetry install --with test,test_integration  # + integration tests
poetry install --with lint,typing       # linting + type checking

# Run all unit tests
make test

# Run a single test file
TEST_FILE=tests/unit_tests/test_chat_models.py make test

# Run a specific test by name
poetry run pytest tests/unit_tests/test_chat_models.py::TestClassName::test_method -v

# Lint (ruff + mypy)
make lint

# Format
make format

# Spellcheck
make spell_check
```

**Before committing**, always run from the package directory:

```bash
make format
make lint_package
make lint_tests
```

Address all issues before pushing. CI runs lint and unit tests only for packages with changed files (see `.github/scripts/check_diff.py`). Tests run on Python 3.10 and 3.12.

## Architecture

### Agent Service Versioning (V1 / V2)

The `langchain-azure-ai` agents module has two parallel implementations:

- **V1** (`agents/_v1/`): Uses `azure-ai-agents` SDK. Public API via `agents/` and `agents/prebuilt/`.
- **V2** (`agents/_v2/`): Uses `azure-ai-projects >= 2.0` (Responses/Conversations API) with OpenAI SDK types. Public API via `agents/v2/` and `agents/v2/prebuilt/`.

The default import path (`from langchain_azure_ai.agents import AgentServiceFactory`) resolves to **V1**. V2 requires explicit import from `langchain_azure_ai.agents.v2`.

Implementation lives in private `_v1/` and `_v2/` directories. Public API directories (`v1/`, `v2/`, `prebuilt/`) only contain `__init__.py` files that re-export via lazy imports.

### Lazy Import Pattern

All `__init__.py` files in submodules use a lazy-import pattern with `_module_lookup` dict and `__getattr__`. This keeps import-time overhead low. When adding new public symbols:

```python
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

### Chat Models and Embeddings

`AzureAIChatCompletionsModel` (in `chat_models/inference.py`) extends LangChain's `BaseChatModel` using the `azure-ai-inference` SDK. It is separate from `AzureChatOpenAI` (re-exported from `langchain-openai`). Both are exposed from `langchain_azure_ai.chat_models`.

## Key Conventions

- **Docstrings**: Google-style (`convention = "google"` in ruff config). Enforced in source but not in tests (`tests/**` ignores `D` rules).
- **Type annotations**: Required on all public functions (`disallow_untyped_defs = true` in mypy).
- **Deprecation/Experimental**: Use decorators from `langchain_azure_ai._api.base` — `@deprecated()` and `@experimental()` — not `langchain_core`. The `@experimental()` decorator emits an Azure-specific preview warning.
- **Optional dependencies**: Heavy SDKs are gated behind extras in `pyproject.toml` (`v1`, `opentelemetry`, `tools`). Guard imports accordingly.
- **Async**: pytest uses `asyncio_mode = "auto"`. Async tests don't need the `@pytest.mark.asyncio` decorator.
- **Network isolation in unit tests**: `pytest-socket` is a test dependency. Unit tests must not make network calls; use mocks and `unittest.mock.patch`.

## MCP Configuration

The repository includes `.mcp.json` at the root with a LangChain docs server (`https://docs.langchain.com/mcp`). This is useful for looking up LangChain API references and guides.

## Release Notes

Each package maintains a `## Changelog` section in its `README.md`. When a new version is released, update that section with a concise, user-friendly summary of what changed.

### How to compile release notes

1. **Identify the two release tags** you are comparing. Tags follow the pattern `<package>==<version>`, for example `langchain-azure-ai==1.2.1`. List available tags with:

   ```bash
   gh release list --repo langchain-ai/langchain-azure
   # or
   git tag --list | grep langchain-azure-ai
   ```

2. **List the PRs merged between the two tags** using the GitHub CLI:

   ```bash
   # Get the commits between the two tags and look for merge commits
   git log <old-tag>..<new-tag> --oneline --merges

   # Or use the GitHub compare API to get a full list of commits
   gh api repos/langchain-ai/langchain-azure/compare/<old-tag>...<new-tag> \
     --jq '.commits[].commit.message'
   ```

   You can also browse the merged PRs on GitHub directly by filtering by merge date range:

   ```
   https://github.com/langchain-ai/langchain-azure/pulls?q=is%3Apr+is%3Amerged+merged%3A<start-date>..<end-date>+label%3A<package-label>
   ```

3. **Write the release notes**. For each meaningful PR (skip internal/CI-only changes), add a bullet using friendly, user-facing language. Follow these conventions:
   - Start sentences with "We" to keep a consistent, team-friendly voice (e.g., "We fixed a problem when loading class X").
   - Use past tense ("We introduced", "We fixed", "We improved").
   - Mark breaking changes with `**[Breaking change]:**`.
   - Mark new features with `**[NEW]**` when the addition is significant.
   - Reference the PR number with a link, e.g., `[#123](https://github.com/langchain-ai/langchain-azure/pull/123)`.

4. **Update the package `README.md`**. Add the new version entry at the top of the `## Changelog` section:

   ```markdown
   ## Changelog

   - **<new-version>**:

       - We fixed an issue with X. [#123](https://github.com/langchain-ai/langchain-azure/pull/123)
       - We introduced support for Y. [#124](https://github.com/langchain-ai/langchain-azure/pull/124)

   - **<previous-version>**:
       ...
   ```

   Each package's `README.md` is the source of truth for its changelog. The files to update are:

   | Package | README location |
   |---------|----------------|
   | `langchain-azure-ai` | `libs/azure-ai/README.md` |
   | `langchain-azure-dynamic-sessions` | `libs/azure-dynamic-sessions/README.md` |
   | `langchain-sqlserver` | `libs/sqlserver/README.md` |
   | `langchain-azure-storage` | `libs/azure-storage/README.md` |
   | `langchain-azure-postgresql` | `libs/azure-postgresql/README.md` |
   | `langchain-azure-cosmosdb` | `libs/azure-cosmosdb/README.md` |
