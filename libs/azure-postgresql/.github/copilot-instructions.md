# langchain-azure-postgresql

This project integrates LangChain with Azure Database for PostgreSQL, providing
seamless access to vector stores and other database functionalities such as,
e.g., Entra ID authentication and DiskANN indexing algorithm, which help improve
security and performance when working with large datasets.

## Tooling

For this project, we use `uv` for packaging and dependency management. Similarly,
we use `pytest` and `tox` to handle testing and test automation. The following
commands/tools are useful for the following purposes:

- **Creating the development environment:** `uv sync --all-extras`
- **Running tests:** `uv run tox -e py`
- **Running lint checks:** `uv run tox -m lint`
- **Running type checks:** `uv run tox -m type`
- **Running packaging checks:** `uv run tox -m package`

If there exists a `.env` file in the root directory, you can replace the
`uv run` command with `uv run --env-file .env` and keep the rest the same.

## Project Structure

The project has the general structure:

```shell
$ tree -L 4 -P '*.py' -P '*.toml'
.
├── pyproject.toml
├── src
│   └── langchain_azure_postgresql
│       ├── common
│       │   ├── aio
│       │   ├── _connection.py
│       │   ├── __init__.py
│       │   └── _shared.py
│       ├── __init__.py
│       └── langchain
│           ├── aio
│           ├── __init__.py
│           ├── _shared.py
│           └── _vectorstore.py
└── tests
    ├── common
    │   ├── __init__.py
    │   ├── test_connection.py
    │   └── test_shared.py
    ├── conftest.py
    ├── __init__.py
    ├── integration_tests
    │   ├── __init__.py
    │   └── test_placeholder.py
    └── langchain
        ├── conftest.py
        ├── __init__.py
        └── test_vectorstore.py

10 directories, 18 files
```

Specifically, the project follows the standard Python package structure, with separate directories for source code (`src/`) and tests (`tests/`), and a
`pyproject.toml` file for all Python-/tooling-related configurations. We aim to
separate the asynchronous code from the synchronous code, ensuring a clear
distinction between the two, under dedicated directories, such as `**/aio/*.py`.

## General Coding Guidelines

Follow the below guidelines when interacting with the project files:

- Be concise,
- Follow the existing code style and conventions,
- Always summarize your plan and ask for confirmation before moving forward,
- Ask before you want to use linters and type checkers to avoid disrupting the
  workflow, and,
- Use docstrings in `sphinx` convention.
