# langchain-azure-compute

LangChain integrations for [Azure Container Apps](https://learn.microsoft.com/azure/container-apps/)
compute that runs agent-authored code.

## Installation

Install only the integration you need:

```bash
pip install "langchain-azure-compute[dynamic-sessions]"
pip install "langchain-azure-compute[sandboxes]"
pip install "langchain-azure-compute[durabletask]"
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

## Durable LangGraph execution (Durable Task Scheduler)

`langchain_azure_compute.graph` runs compiled LangGraph graphs (`StateGraph.compile()`),
LangChain agents (`create_agent` / `create_react_agent`), and Deep Agents
(`create_deep_agent`) on the
[Azure Durable Task Scheduler (DTS)](https://learn.microsoft.com/azure/azure-functions/durable/durable-task-scheduler/durable-task-scheduler)
or the local DTS emulator.

```python
from typing import Annotated, TypedDict

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from langchain_azure_compute.graph import attach_durability, durable_tool, sdk_options


class SupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    approved: bool


@durable_tool(retries=3)
@tool
def refund_order(order_id: str) -> str:
    """Runs as an isolated Durable Task activity with its own retry policy."""
    return f"Refunded order {order_id}"


def manager_gate(state: SupportState) -> dict:
    decision = interrupt({"question": "Approve refund?", "order_id": "ord-42"})
    return {
        "approved": bool(decision.get("approved")),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "refund_order",
                        "args": {"order_id": "ord-42"},
                    }
                ],
            )
        ],
    }


builder = StateGraph(SupportState)
builder.add_node("approve", manager_gate)
builder.add_node("tools", ToolNode([refund_order]))
builder.add_edge(START, "approve")
builder.add_conditional_edges(
    "approve",
    lambda s: "tools" if s.get("approved") else END,
    {"tools": "tools", END: END},
)
builder.add_edge("tools", END)
graph = builder.compile()

reg = attach_durability(graph, name="support")

# Connect to the local DTS emulator ("127.0.0.1:8080") or an Azure DTS endpoint
# ("https://<scheduler>.<region>.durabletask.io") with DefaultAzureCredential:
opts = sdk_options("127.0.0.1:8080")
worker = DurableTaskSchedulerWorker(**opts)
sdk = DurableTaskSchedulerClient(**opts)
reg.register(worker)
worker.start()
try:
    client = reg.client(sdk)
    run = client.start(
        {"messages": [HumanMessage(content="Refund order 42")], "approved": False},
        instance_id="refund-42",
    )
    pending = run.wait_pending(timeout=30)
    run.resume(pending, {"approved": True})
    result = run.wait_result(timeout=60)
finally:
    worker.stop()
    sdk.close()
```

Key capabilities include:

- **Control plane vs. state plane separation:** With a persistent checkpointer
  (`SqliteSaver`, `PostgresSaver`, or `CosmosDBSaver`), step transitions pass
  `< 1 KB` checkpoint IDs instead of full checkpoint bundles. Durable tool/node
  inputs and outputs, plus final output returned through DTS, still travel in
  DTS payloads. Zero-config `InMemorySaver` checkpoints also travel as inline
  bundles. Long-running conversations roll over via `continue_as_new`.
- **Per-tool and per-node activity isolation:** `@durable_tool` calls inside a
  `ToolNode` and `@durable_node` branches (including dynamic `Send()` map-reduce
  fan-out) execute as parallel DTS activities with independent retry policies.
- **Human-in-the-loop (`interrupt()` / `resume()`):** Maps LangGraph `interrupt()`
  and static breakpoints to DTS external events without holding worker threads.
- **Azure Functions hosting (`LangGraphFunctionApp`):** Exposes `/api/graphs/<name>/start`,
  `/status/{instanceId}`, and `/respond/{instanceId}/{interruptId}` routes on Azure
  Functions Python v2.

  Function apps require verified per-instance identity through Azure
  EasyAuth or an `identity_provider` token-validation callback. A Function key alone
  does not identify a user. This also applies to `AuthLevel.ANONYMOUS` behind
  EasyAuth: gateway authentication does not disable instance ownership checks.
  ANONYMOUS with no identity source requires explicit `require_instance_owner=False`;
  omission raises `ValueError`. False explicitly selects shared,
  platform-key-authorized access without user isolation. See the
  [Functions sample identity contract](../../samples/langgraph-durabletask/10_azure_functions_hitl/README.md#identity-and-ownership-contract)
  for deployment requirements and the opt-in loopback demo.

#### Supported Functions and Worker Topologies

- **Topology A (recommended):** Azure Functions, ACA, or App Service serves thin
  HTTP control routes with `start_worker=False` (the default on every helper).
  Dedicated `DurableTaskSchedulerWorker` replicas run on ACA or AKS, scaled by
  the `azure-durabletask-scheduler` KEDA scaler, or on a Dedicated App Service plan
  with Always On. The HTTP host and all workers must use the same scheduler and hub.
- **Topology B (local development / Dedicated Always-On only):**
  explicitly set `start_worker=True` for single-process `func start`, local Python
  testing, or a Dedicated plan with Always On enabled.

An embedded worker is **not supported on Consumption or Flex**. It is not an
Azure Functions trigger: the scale controller cannot see DTS activities, may kill
the host mid-activity on scale-in, and can scale to zero after `/respond` returns,
leaving queued work stalled. A successful HTTP response means the event was queued,
not that a graph worker is alive.

By default `/respond` is owner-only. Supply
`respond_authorizer(caller, instance_id, owner[, graph_name])` (legacy
two-argument callbacks are also supported) to require an independently
authenticated approver, for example a configured manager principal distinct from
the starter. Authorized approvers can read `/status` before responding. Ownership
and graph-scope lookups consult DTS and fail closed; an instance from another
graph cannot be controlled through these routes. The server overwrites dict resume
fields `responder` and `reviewer` with the verified caller. For a batch
`{"value": {"interrupt-id": {...}}}`, it stamps each dict payload, never the outer
interrupt-ID map. The expense sample uses this identity, not a body-supplied reviewer.

Authenticated default thread IDs are owner-namespaced. Checkpoints record
`durable_owner` alongside `durable_instance_id`; sync/async starts and worker
steps reject checkpoints belonging to another principal even after DTS purge.
Explicit `thread_id` remains supported by the Python API, not by HTTP `/start`.
HTTP bodies are limited to 1,000,000 bytes, instance IDs to
`[A-Za-z0-9_.:@-]{1,128}`, and internal resume envelope keys are rejected.
Ordinary user fields such as `{"approved": true, "version": 2}` remain valid
interrupt response data.

### Execution Guarantees, Streaming, Tracing & Data Retention

- **At-least-once activities:** Durable tools and nodes may execute more than
  once, including after a worker crashes between an external mutation and its
  completion acknowledgement. Completed history avoids ordinary replay of
  completed siblings, but does not provide exactly-once external effects.
  Tools receive a deterministic `runtime.config["metadata"]["durable_call_id"]`
  for use as an external idempotency key. Persist deduplication at the mutation
  boundary (provider API or a database unique constraint), not just in RAM.
- **Streaming and tracing:** `stream_writer` inside a durable tool activity is
  a no-op. This adapter does not expose a continuous token stream over DTS.
  LangSmith traces are scoped to individual supersteps/activities rather than
  one uninterrupted graph invocation.
- **DTS retention:** Tool/node activity inputs and outputs are recorded in DTS
  history. Zero-config `InMemorySaver` checkpoints and final graph state also
  travel in DTS payloads. Interrupt payloads whose serialized custom-status
  representation fits the 16 KB limit may be inlined; larger values use checkpoint
  references. Review sensitive data, DTS
  retention and purge policies, and checkpoint-store retention independently.
- **Deployed Azure auth:** Prefer `Authentication=ManagedIdentity`, with
  `ClientID=<user-assigned-client-id>` when using a user-assigned identity.
  Grant that identity the **Durable Task Data Contributor** RBAC role on the
  intended scheduler/task hub. Developer `AzureCli` credentials are not a
  production authentication strategy.

### Connecting to Hosted Azure DTS (Entra ID / Managed Identity) vs. Local Emulator

`sdk_options()` supports both the local DTS emulator and hosted Azure Durable Task
Scheduler resources:

```python
import os
from azure.identity import DefaultAzureCredential
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langchain_azure_compute.graph import sdk_options

# 1. Local DTS Emulator (secure_channel=False, token_credential=None, taskhub="default")
emulator_opts = sdk_options("127.0.0.1:8080")

# 2. Hosted Azure DTS with Entra ID / Managed Identity (secure_channel=True auto-enabled)
hosted_opts = sdk_options(
    "https://<scheduler>.<region>.durabletask.io",
    taskhub="my-taskhub",
    token_credential=DefaultAzureCredential(),
)

# 3. Hosted Azure DTS via Connection String (Endpoint=...;TaskHub=...;Authentication=DefaultAzure)
conn_str_opts = sdk_options(
    connection_string=os.environ["DURABLE_TASK_SCHEDULER_CONNECTION_STRING"],
    token_credential=DefaultAzureCredential(),
)

worker = DurableTaskSchedulerWorker(**hosted_opts)
sdk = DurableTaskSchedulerClient(**hosted_opts)
```

### Two-Tier State Model

| Mode | How to Enable | How State Moves | Best For |
| --- | --- | --- | --- |
| **Tier 1: Shared Checkpointer** *(Checkpoint IDs)* | `builder.compile(checkpointer=SqliteSaver(...))` (compatible with `SqliteSaver`, `AsyncSqliteSaver`, or any `BaseCheckpointSaver`) | `inline_checkpoints=False` (default). Step transitions pass `< 1 KB` checkpoint IDs instead of full checkpoint bundles stored in your external database. Durable tool/node inputs and outputs, plus final output returned through DTS, still travel in DTS payloads. | Multi-turn conversations across stateless API pods, large state payloads, long-lived threads. |
| **Tier 2: Zero-Config** *(Inline Checkpoint Bundle)* | `builder.compile()` (no checkpointer or `InMemorySaver()`) | `inline_checkpoints=True` (default). After each step, the latest checkpoint slice (up to the 900 KB bundle guard) is exported into `checkpoint_bundle` and hydrated into the next worker's `InMemorySaver`. | Single-turn runs or getting started immediately with zero database setup while still surviving worker crashes and multi-worker routing. |

Final inline results carry the checkpoint bundle once, without a duplicate state
blob; `wait_result()` reads the projected output from the hydrated checkpoint.
Every complete activity input/result, orchestration input/output, and
`continue_as_new` input must fit the 950,000-byte serialized payload guard. The
900 KB bundle guard is a separate limit on the encoded bundle, not raw graph state.
Oversized payloads fail without activity retries. Use a persistent checkpointer
for large inline checkpoint bundles. For large durable tool/node inputs or return
values, reduce their size or store blobs externally and pass references.
`sdk_options(payload_store=...)` does not bypass the integration's wire-size guard.

Caller-supplied checkpointer serializers are preserved, including custom or
encrypted `InMemorySaver` serializers. For custom state types used in activity
payloads, register explicitly trusted types with
`attach_durability(..., allowed_msgpack_modules=[("myapp.models", "MyState")])`.
This allowlist is process-wide; configure it on both clients and workers.
`allowed_msgpack_modules=True` permits arbitrary types and should only be used
with trusted payloads.

Async checkpointers retain their caller-owned event loop and lock.
`Limits(async_timeout_seconds=300.0)` controls bridged async operation timeouts;
use `None` to disable the timeout. Async clients read state directly on their
current event loop.

Concurrent starts from the same checkpoint are fenced by the recorded run ID and
input signature. A different run or input raises `RunError` instead of silently
replaying another turn. Default run IDs are deterministic for the graph name,
thread, base checkpoint, and input; sequential turns with new checkpoints get
distinct IDs. Wait for an active turn to finish before submitting another turn.
Pending interrupt status is explicitly cleared after a response and on completion;
terminal runs never return pending receipts. Owner identity stays in the durable
orchestration input, not public custom status or output.

`sdk_options()` resolves task hubs from explicit arguments, connection strings,
`DTS_TASKHUB`, then `TASKHUB_NAME`. Credentials cannot be sent to a non-loopback
host with `secure_channel=False`. `max_message_bytes` is optional and does not
change the SDK's gRPC defaults unless supplied.

### Configuring Tool Durability, Retries, and Concurrency

By default (`durable_tools=True`), every tool inside a `ToolNode` executes as an
isolated `langgraph.<name>.tool` activity. You can customize retries or opt out
pure in-memory helpers either via `@durable_tool` or centrally on `attach_durability()`:

Durable tool dispatch follows the actual scheduled LangGraph tasks, including
`Send()` routing: HITL-rejected or human-responded calls are not executed.
Invalid tool arguments handled by `ToolNode` produce an error `ToolMessage`
immediately, allowing the model to self-correct without activity retries.
Other tool errors retain retries; when exhausted, custom error policies are
evaluated against the original exception type before crossing the DTS boundary.

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_azure_compute.graph import attach_durability, durable_tool


@durable_tool(retries=4)
@tool
def book_flight(destination: str) -> str:
    """Runs as an isolated DTS activity with up to 4 attempts."""
    return f"FLIGHT-{destination}"


@durable_tool(durable=False)
@tool
def format_itinerary(summary: str) -> str:
    """Pure in-memory helper: runs locally inside the tool-commit step."""
    return summary.upper()


reg = attach_durability(
    graph,
    name="travel_agent",
    tool_policy={"tools::book_flight": "durable", "tools::format_itinerary": "local"},
    tool_retries={"book_flight": 4},
    tool_max_concurrency=2,  # None = parallel batch (default), 1 = sequential, N = batches of N
)
```

### Durable Nodes (`@durable_node`) for Parallel Fan-Out & `Send()` Map-Reduce

When a graph fans out across parallel branches (`START -> [branch_a, branch_b] -> join`)
or uses `Send("worker", item)` map-reduce, decorate the branch/target node with
`@durable_node(retries=3)` (or pass `durable_nodes=True` / `node_policy=...` to
`attach_durability()`). Each parallel branch or `Send()` packet executes concurrently
as its own `langgraph.<name>.node` activity across workers with per-branch retry
isolation.

### Graph Versioning & Live Migration

When deploying an updated graph topology (`"v2"`) while `"v1"` runs are in flight:

```python
reg = attach_durability(
    graph_v2,
    name="order_workflow",
    version="v2",
    previous_versions={"v1": graph_v1},
    migrations={
        ("v1", "v2"): lambda state, snap, resume_val: {
            "as_node": "new_approval_gate",
            "values": {"approved_by": resume_val},
        }
    },
    upgrade_on_continue_as_new=True,
)
```

### Runnable Samples

Ten end-to-end runnable samples (each with its own `README.md`, `requirements.txt`,
and `sample.py`) are available in [`samples/langgraph-durabletask/`](../../samples/langgraph-durabletask/README.md):

| # | Sample | What It Demonstrates |
| --- | --- | --- |
| 01 | [`01_hello_graph`](../../samples/langgraph-durabletask/01_hello_graph/README.md) | Minimal linear & conditional graph (Zero-Config `builder.compile()`) |
| 02 | [`02_vacation_booking`](../../samples/langgraph-durabletask/02_vacation_booking/README.md) | Per-tool retry isolation inside a `ToolNode` (`book_car` retries without re-running `book_flight` or `book_hotel`) & idempotent `client.start(..., instance_id=...)` |
| 03 | [`03_human_in_the_loop`](../../samples/langgraph-durabletask/03_human_in_the_loop/README.md) | Sequential and parallel `interrupt()`, `pending_interrupts()`, and `run.resume()` |
| 04 | [`04_react_agent_and_subgraphs`](../../samples/langgraph-durabletask/04_react_agent_and_subgraphs/README.md) | Prebuilt ReAct agents (`create_agent` / `create_react_agent`) and compiled subgraphs as nodes |
| 05 | [`05_multi_turn_conversation`](../../samples/langgraph-durabletask/05_multi_turn_conversation/README.md) | Multi-turn conversation with `SqliteSaver`, `SqliteStore`, and `Runtime[Context]` propagation |
| 06 | [`06_durable_nodes_map_reduce`](../../samples/langgraph-durabletask/06_durable_nodes_map_reduce/README.md) | Static parallel fan-out and dynamic `Send()` map-reduce with `@durable_node` and `node_max_concurrency` |
| 07 | [`07_multi_graph_app`](../../samples/langgraph-durabletask/07_multi_graph_app/README.md) | Multiple durable graphs (`triage`, `vacation`, `refund`) sharing one worker and delegating across graphs |
| 08 | [`08_graph_versioning_and_migration`](../../samples/langgraph-durabletask/08_graph_versioning_and_migration/README.md) | Graph versioning and live checkpoint migration (`version="v2"`, `previous_versions`, `migrations`, `run.upgrade()`) |
| 09 | [`09_two_process_crash_recovery`](../../samples/langgraph-durabletask/09_two_process_crash_recovery/README.md) | Two OS worker processes (`pod-A` hard-killed mid-run, `pod-B` recovers via Zero-Config `checkpoint_bundle`) |
| 10 | [`10_azure_functions_hitl`](../../samples/langgraph-durabletask/10_azure_functions_hitl/README.md) | Azure Functions Python v2 HTTP routes (`LangGraphFunctionApp` `/start`, `/status/{id}`, `/respond/{id}/{interruptId}`) across two worker processes |

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

## Changelog

- **0.2.0**:

  - This is the first release of `langchain-azure-compute` published to PyPI. It includes all changes listed under 0.1.1 below.
  - **[NEW]** We introduced the `langchain-azure-compute` package for running LangChain, LangGraph, and Deep Agents workloads on Azure compute services. [#907](https://github.com/langchain-ai/langchain-azure/pull/907)
  - **[NEW]** We added the Azure Container Apps dynamic sessions tools under `langchain_azure_compute.dynamic_sessions`. [#908](https://github.com/langchain-ai/langchain-azure/pull/908)
  - **[NEW]** We added a Deep Agents backend that runs code and file operations inside Azure Container Apps dynamic sessions. [#909](https://github.com/langchain-ai/langchain-azure/pull/909)
  - **[NEW]** We added `ACASandbox`, a Deep Agents sandbox backend for Azure Container Apps. [#910](https://github.com/langchain-ai/langchain-azure/pull/910)
  - **[NEW]** We added durable execution for LangGraph graphs and LangChain agents with Azure Durable Task Scheduler under `langchain_azure_compute.graph`, available through the `durabletask` extra. It supports durable tools and nodes, human-in-the-loop interrupts, graph versioning and migration, and Azure Functions hosting. [#1270](https://github.com/langchain-ai/langchain-azure/pull/1270)

- **0.1.1**:

  - We raised the minimum supported Python version from 3.10 to 3.11 in accordance with the repository's Python support policy. Users running Python 3.10 must upgrade their runtime to install this release. [#1021](https://github.com/langchain-ai/langchain-azure/pull/1021)
