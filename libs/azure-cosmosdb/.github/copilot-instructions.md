# langchain-azure-cosmosdb

This project provides Azure CosmosDB NoSQL integrations for both LangChain and
LangGraph, including vector store, semantic cache, chat message history,
query constructors, LangGraph checkpointer, and LangGraph cache — all with
both synchronous and asynchronous implementations.

## Tooling

For this project, we use `poetry` for packaging and dependency management, and
`pytest` for testing. The following commands are useful:

- **Creating the development environment:** `poetry install --with test,lint,typing`
- **Running unit tests:** `make test`
- **Running integration tests:** `make integration_tests`
- **Running lint checks:** `make lint_package` and `make lint_tests`
- **Formatting code:** `make format`

If there exists a `.env` file in the root directory, source it before running
integration tests: `set -a && source .env && set +a`.

## Project Structure

The project has the general structure:

```shell
$ tree -L 4 -P '*.py' --prune
.
├── pyproject.toml
├── src
│   └── langchain_azure_cosmosdb
│       ├── __init__.py
│       ├── langchain
│       │   ├── __init__.py
│       │   ├── _cache.py
│       │   ├── _chat_history.py
│       │   ├── _query_constructor.py
│       │   ├── _utils.py
│       │   ├── _vectorstore.py
│       │   └── aio
│       │       ├── __init__.py
│       │       ├── _cache.py
│       │       ├── _chat_history.py
│       │       └── _vectorstore.py
│       └── langgraph
│           ├── __init__.py
│           ├── _cache.py
│           ├── _checkpoint_store.py
│           └── aio
│               ├── __init__.py
│               ├── _cache.py
│               └── _checkpoint_store.py
└── tests
    ├── __init__.py
    ├── unit_tests
    │   ├── __init__.py
    │   ├── test_imports.py
    │   ├── langchain
    │   │   └── (sync + async tests)
    │   └── langgraph
    │       └── (sync + async tests)
    └── integration_tests
        ├── __init__.py
        ├── langchain
        │   └── (sync + async tests)
        └── langgraph
            └── (sync + async tests)
```

Specifically, the project follows the standard Python `src/` layout, with
separate directories for LangChain integrations (`langchain/`) and LangGraph
integrations (`langgraph/`). Asynchronous code lives under dedicated `aio/`
subdirectories.

## General Coding Guidelines

Follow the below guidelines when interacting with the project files:

- Be concise,
- Follow the existing code style and conventions,
- Always summarize your plan and ask for confirmation before moving forward,
- Use Google-style docstrings (configured via ruff pydocstyle),
- Ensure all public functions have type annotations, and,
- Keep sync and async implementations in separate files.
