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

The Deep Agents backend is marked `@beta`: deepagents is pre-1.0, so this
interface may change.

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

### Deep Agents backend

`SessionsBashBackend` implements the Deep Agents `SandboxBackendProtocol`
against a **Shell-typed** session pool. Its file operations are bash-native
rather than the `python3 -c` wrappers `BaseSandbox` uses, which a shell-only
pool image may not be able to run.

```python
from deepagents import create_deep_agent

from langchain_azure_compute.dynamic_sessions.backends import (
    SessionsBashBackend,
)

backend = SessionsBashBackend(pool_management_endpoint="<pool-management-endpoint>")
agent = create_deep_agent(model="...", backend=backend)
```

`upload_files`/`download_files` use the session's file API, which is a flat
store rooted at `/mnt/data`. Only `/mnt/data/<name>` is storable; any other
path is rejected with `invalid_path` rather than silently written elsewhere.
`write()` has no such limit — it goes through the shell and can create any
path.

The session data plane intermittently drops a command's final line of output
(measured at 1.6–4%). `ls`, `read`, `glob` and `grep` detect this and retry,
so they never report a short result as a complete one. **`execute()` is not
covered**: it runs your command verbatim, and wrapping it would change the
exit status it returns. If you parse the last line of `execute()` output,
terminate the command with a marker of your own and re-run when it is missing.

The data plane also caps each command's stdout and stderr at 4,096 bytes,
silently. `read` pages beneath the cap in base64 chunks, so any file within
`max_output_bytes` reads intact — larger windows just cost more round trips.
A listing, glob, or grep whose output exceeds the cap says so: `ls` advises
listing a subdirectory, `glob`/`grep` return what fits, flagged `truncated`.
`execute()` sets `truncated=True` on a stream arriving at the cap — a
heuristic, and the only signal there is, since the service reports none.

Three behavioral notes:

- `write()` refuses to overwrite an existing file, matching the shared
  `langchain-tests` sandbox suite; the error tells the model to use `edit` or
  delete first.
- `write()` content and `edit()` strings ride inside a single shell command,
  which Linux caps at 128 KiB per string; payloads above ~90 KB are refused
  with an error saying so. `edit()` budgets `old_string` and `new_string`
  together, because both travel in that one command. Larger transfers belong
  in `upload_files`, within its `/mnt/data` limit.
- `glob()` follows the protocol's Python-glob semantics for the common shapes
  (`*.py` stays in one directory level, `**/name` recurses); patterns mixing
  several `**` segments take a documented `find -path` approximation. As in
  Python, a wildcard does not match a leading dot — pass a pattern that names
  the hidden entry (`.env`, `.github/**`) to reach one. Naming a hidden
  directory opens that directory only, not hidden entries beneath it.

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
| `AZURE_DYNAMIC_SESSIONS_SHELL_POOL_MANAGEMENT_ENDPOINT` | `SessionsBashBackend`, against a **Shell**-typed pool | `Azure ContainerApps Session Executor` on the pool |
