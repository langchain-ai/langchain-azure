# langchain-azure-compute

LangChain integrations for [Azure Container Apps](https://learn.microsoft.com/azure/container-apps/)
compute that runs agent-authored code.

## Installation

Install only the integration you need:

```bash
pip install "langchain-azure-compute[dynamic-sessions]"
```

Extras control *dependencies*, not which modules are present — every module
ships in the wheel. Each subpackage checks for its own requirements at import
time and raises an error naming the extra to install.

## Dynamic sessions

### Tools

`SessionsPythonREPLTool` and `SessionsBashTool` run LLM-authored code in an
ephemeral session and return its output.

```python
from langchain_azure_compute.dynamic_sessions import SessionsPythonREPLTool

tool = SessionsPythonREPLTool(pool_management_endpoint="<pool-management-endpoint>")
print(tool.invoke("print(sum(range(10)))"))
```

By default a `DefaultAzureCredential` fetches a token for the
`https://dynamicsessions.io` scope; pass `access_token_provider=` to supply one
yourself. Requires the `Azure ContainerApps Session Executor` role on the
session pool.

## Development

See [CONTRIBUTING.md](../../CONTRIBUTING.md) in the repository root.

```bash
make test          # unit tests
make lint          # ruff + mypy
make check_imports # import smoke check
```

### Running the integration tests

These run against live Azure resources. Each suite skips itself when its
variables are unset, so `make integration_tests` is safe to run with nothing
configured — that is what release CI does.

```bash
cp .env.example .env   # then fill in the resources you have
az login
make integration_tests
```

`.env` is gitignored and loaded automatically. Authentication is
`DefaultAzureCredential`, so `az login` is enough; no credential belongs in the
repository.

| Variable | Suite | Role |
|---|---|---|
| `AZURE_DYNAMIC_SESSIONS_POOL_MANAGEMENT_ENDPOINT` | the tools, against a **Python**-typed pool | `Azure ContainerApps Session Executor` on the pool |
