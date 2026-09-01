# langchain-azure-compute

LangChain integrations for [Azure Container Apps](https://learn.microsoft.com/azure/container-apps/)
compute that runs agent-authored code.

## Installation

Install only the integration you need:

```bash
pip install "langchain-azure-compute[dynamic-sessions]"
pip install "langchain-azure-compute[sandboxes]"
```

Extras control *dependencies*, not which modules are present — every module
ships in the wheel. Each subpackage checks for its own requirements at import
time and raises an error naming the extra to install.

Both Deep Agents backends are marked `@beta`: deepagents is pre-1.0 and ACA
sandboxes is an Early Access service, so these interfaces may change.

## Which one do I want?

Dynamic sessions and sandboxes are **different Azure products**, not tiers of
one. They use different ARM resource types and different data planes.

| | Dynamic sessions | Sandboxes |
|---|---|---|
| ARM resource | `Microsoft.App/sessionPools` | `Microsoft.App/sandboxGroups` |
| Access | HTTP via a session pool endpoint | per-sandbox data plane SDK / `aca` CLI |
| State | ephemeral, destroyed after cooldown | stateful: suspend, resume, snapshots |
| Persistent storage | none | volumes (Azure Blob, Data Disk) |
| Networking | basic isolation | egress policies, VNet integration, port management |
| Best for | one-shot LLM-generated code execution | long-running agent workspaces, dev environments |

Choose **dynamic sessions** for a managed execution experience that abstracts
away infrastructure. Choose **sandboxes** when you need programmable control
over isolated compute that keeps state across tasks.

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

## Sandboxes: Deep Agents backend

`ACASandbox` implements the Deep Agents `SandboxBackendProtocol` on top of an
Azure Container Apps sandbox. It wraps a client you construct, so it never
builds endpoints or handles credentials itself, and you keep control of the
sandbox lifecycle through that same client.

```python
from azure.containerapps.sandbox import SandboxClient, endpoint_for_region
from azure.identity import DefaultAzureCredential
from deepagents import create_deep_agent

from langchain_azure_compute.sandboxes import ACASandbox

client = SandboxClient(
    endpoint_for_region("westus2"),
    DefaultAzureCredential(),
    subscription_id="<subscription-id>",
    resource_group="<resource-group>",
    sandbox_group="<sandbox-group>",
    sandbox_id="<sandbox-id>",
)
client.ensure_running()

agent = create_deep_agent(model="...", backend=ACASandbox(client))
```

Requires the `Container Apps SandboxGroup Data Owner` role on the sandbox
group.

Two options are worth knowing about:

- `async_client=` — pass `azure.containerapps.sandbox.aio.SandboxClient` for the
  same sandbox to run async operations on the SDK's async transport. Without
  it, they run the sync client in a worker thread. Either way `aread`,
  `awrite`, `aupload_files` and `adownload_files` take the same SDK path as
  their sync counterparts rather than the base class's shell implementation --
  except that a read above the ~10 MiB routing cap deliberately falls back to
  the base class's server-side path, as `read`/`aread` both do.
  Call `aclose()` when done to release the async client's connection pool.
- `enable_capture_offload=True` — offloads large command output at the source.
  Needs a POSIX shell and coreutils in the disk image: fine for the `ubuntu`,
  `debian`, and `python` presets, not guaranteed for `alpine` or BYO OCI
  images, so it is off by default.

`SandboxClient.exec()` has no server-side command timeout, so `execute(...,
timeout=N)` is honored by wrapping the command in coreutils `timeout`. On an
image without it, the command still runs — untimed.

The HTTP request itself is bounded by the client transport's read timeout
(about 300 seconds on the SDK default), and a command that outruns it fails
the request first: `execute` then returns an error `ExecuteResponse` rather
than raising, and the command may keep running server-side. To use timeouts
above ~300 seconds, construct the `SandboxClient` with a transport whose read
timeout exceeds them.

## Changelog

- **0.1.0**:

  - We introduced Python and Bash tools for running agent-authored code in Azure Container Apps dynamic sessions. [#908](https://github.com/langchain-ai/langchain-azure/pull/908)
  - We added a Deep Agents backend for persistent shell execution and file operations in dynamic sessions. [#909](https://github.com/langchain-ai/langchain-azure/pull/909)
  - We added the `ACASandbox` Deep Agents backend for stateful Azure Container Apps sandbox environments. [#910](https://github.com/langchain-ai/langchain-azure/pull/910)

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
| `AZURE_SUBSCRIPTION_ID`, `AZURE_CONTAINER_APPS_RESOURCE_GROUP`, `AZURE_CONTAINER_APPS_SANDBOX_GROUP`, `AZURE_CONTAINER_APPS_REGION` | `ACASandbox`, all four required together | `Container Apps SandboxGroup Data Owner` on the sandbox group |

The sandbox suite provisions and deletes a real sandbox per test class, so it
carries an `aca_sandbox` marker:

```bash
uv run --frozen --all-extras --group test --group test_integration \
    pytest tests/integration_tests -m "not aca_sandbox"
```
