# Sample 99 — Resilient background Responses + LangGraph checkpointer

> **Work in progress / experimental.** This sample is the sandbox for
> integrating the **resilient background responses** feature of
> `azure-ai-agentserver-responses` with LangGraph's native checkpointer,
> and for the corresponding changes to
> `langchain_azure_ai.agents.hosting.ResponsesHostServer`.

## What this sample demonstrates

This is a real-model trip-planning agent built as a
[LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` and hosted
over the **Responses protocol**. Flight and hotel searches run automatically,
but the sensitive `book_trip` tool is blocked by a durable `interrupt()` until
the client sends an explicit approval.

```text
START -> agent -> search tools -> agent -> approval -> book_trip -> agent -> END
```

`ResponsesHostServer` emits the pause as an `mcp_approval_request`. The Textual
client displays the tool arguments and enables **Approve** and **Deny**.

### Progression (all done)

1. **[done] Baseline** — deterministic numeric pipeline, running locally.
2. **[done] Enable resilience** — `ResponsesServerOptions(resilient_background=True)`
   plus a persistent LangGraph checkpointer (`AsyncSqliteSaver`) so graph
   state survives a process restart.
3. **[done] Recovery-aware hosting** — `ResponsesHostServer.handle_create`
   resumes from the exact persisted LangGraph `thread_id` and `checkpoint_id`,
   seeds the reconnecting stream from `context.persisted_response`, and runs
   only the graph work after that native checkpoint.
4. **[done] Exact checkpoint persistence** — each LangGraph checkpoint stream
   event persists its exact `thread_id` and `checkpoint_id` on the task's
   `internal_metadata`. Successful turns also update the linear conversation
   chain's latest exact checkpoint reference.
5. **[done] Crash-recovery test** — killing the server mid-turn and
   restarting resumes the response to completion with an identical output.
6. **[done] Async CUI** — a Textual client streams, reconnects, cancels, and
   automatically steers when a message is sent during active output.
7. **[done] Human approval** — `book_trip` executes only after a structured
   Responses approval resumes the saved LangGraph interrupt.

### Scope

Supports `background=true` + `resilient_background=true` + `stream=true`.
Steerable conversations can be enabled per deployment with
`STEERABLE_CONVERSATIONS=true`.

## Resilience contract

Enabling `resilient_background=True` is only the hosting switch. It changes a
stored background response from "fail after a host crash" to "re-invoke the
handler after the host restarts." Application code must be designed for that
re-invocation.

### Client responsibilities

- Send resilient work with both `background=true` and `store=true`.
   `background=true` with `store=false` is invalid because there is no durable
   response to recover.
- Save the `response.id` as soon as it is received. After a disconnect, poll or
   reconnect to that same response; do not submit the create request again.
- Treat transport loss as an unknown execution state, not as failure. The
   response may still be running and may complete after the host restarts.
- Continue conversations linearly. This integration persists one latest exact
   LangGraph checkpoint per conversation chain; it does not support forking an
   older response into a second branch.

The sample CUI implements these rules: it keeps the response ID, reconnects the
SSE stream after its last sequence number with resumable streaming GET, and
owns the parent response internally.

### Graph and handler responsibilities

- Compile the graph with a durable checkpointer. `ResponsesHostServer` rejects
   `resilient_background=True` when the graph has no checkpointer. The
   checkpointer must survive process replacement and be accessible to every
   host instance that can recover the response.
- Keep all durable workflow state in LangGraph state. Module globals, process
   memory, local caches, temporary files, `ResponseContext`, and cancellation
   events are transient and are reconstructed or lost after restart.
- Make graph nodes replay-safe. There is an at-least-once window between a
   LangGraph checkpoint commit and the matching Responses checkpoint commit. A
   crash in that window resumes from the previously paired checkpoint, so the
   last node can execute again.
- Make external side effects idempotent or deduplicate them with a stable
   operation key stored in graph state. This includes writes, payments, email,
   queue publication, tool calls, and calls to systems that mutate state.
   Recording only "completed" after a side effect is not sufficient: the
   process can crash after the side effect succeeds but before that state is
   checkpointed.
- Keep checkpointed state serializable and compatible across deployments. A
   recovered response may load state written by the previous application
   version.
- Treat cancellation as a request to stop future work, not as a rollback.
   Checkpoints and external effects committed before cancellation remain.

The graph definition may be recreated on every process start; the workflow
must not depend on the identity or memory of the process that created it. In
that sense, graph execution should be stateless even though its durable
workflow state is explicitly checkpointed.

### Production checklist

Before enabling resilience for a real agent, crash-test it at every node
boundary and immediately before and after each external side effect. Verify
that recovery produces one logical result, duplicate effects are suppressed,
the checkpointer survives replacement of the host process, and the client can
recover using only its stored response ID.

## How It Works

### TODO checklist and model modes

The graph keeps this checklist in LangGraph state:

```text
- [ ] 1. Plan the work
- [ ] 2. Research the details
- [ ] 3. Execute the plan
- [ ] 4. Summarize the result
```

Planning resets turn-local TODO and routing state on every fresh turn. This is
important for steering: LangGraph starts the new invocation at `START` using
the parent checkpoint's conversation state, so `START` does not imply empty
state. Use `skip research`, `skip execute`, `research only`, or `summarize only`
in the input to exercise optional routes; skipped phases render as `[-]`.

Each executed node checks one item and emits the full checklist. The graph runs with
LangGraph `durability="sync"`, so the checkpoint for a completed phase is
persisted before the next phase starts. When the request input contains
`crash`, the process deliberately kills itself at the start of the second
phase (`research`). On recovery, LangGraph resumes from the checkpointed state:
the plan TODO is already checked, and the recovered process checks the research
and summarize TODOs.

By default, `FakeChatModel` emits the phase text deterministically, which keeps
crash-recovery demonstrations reproducible. With a model deployment configured,
the same phase text is included in a prompt to the Foundry model; graph routing,
TODO state, crash behavior, and checkpoint boundaries remain deterministic.

The output is intended to make both layers visible:

- **Responses checkpointing** — the stream shows text emitted before the crash,
   then the CUI shows a reconnecting state and resumes the same stored response
   after its last received SSE `sequence_number`.
- **LangGraph checkpointing** — the TODO list after recovery already contains
  `- [x] 1. Plan the work`, proving graph state survived the process crash.

### Responses hosting

`langchain_azure_ai.agents.hosting.ResponsesHostServer` exposes the
compiled graph through the Responses protocol. Resilience is opt-in:

```python
options = ResponsesServerOptions(resilient_background=True)
await ResponsesHostServer(graph, options=options).run_async(port=port)
```

The sample client omits `conversation` and continues the interactive session
with `previous_response_id`. The host stores the stable LangGraph thread ID and
latest checkpoint ID in Agent Server's public `conversation_chain_metadata`
namespace, so each turn continues the latest checkpoint without reading
Responses storage.

Checkpointed conversations are linear. The client owns the parent response
internally; it is not exposed as a command-line option. When steering is
enabled, sending during active output immediately creates a replacement
response whose `previous_response_id` is the active response.

The graph is compiled with a persistent checkpointer so state survives a
restart:

```python
async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
    graph = _build_graph(checkpointer)
    ...
```

`AsyncSqliteSaver` (not the sync `SqliteSaver`) is required because the host
drives the graph with the async API (`astream` / `aget_state`).

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHECKPOINT_DB` | `checkpoints.sqlite` (cwd) locally; `$HOME/checkpoints.sqlite` when hosted | SQLite file backing the LangGraph checkpointer. On Foundry the working directory is ephemeral and lost on restart, so the DB defaults to `$HOME` — the only persisted path — otherwise crash recovery would have no state to resume from. An explicit value always wins. Foundry is detected via `FOUNDRY_HOSTING_ENVIRONMENT`. |
| `AGENTSERVER_STATE_ROOT` | `~/.agentserver` | Root of the local file-backed resilient task/response/stream stores. |
| `TOKEN_DELAY_SECONDS` | `0.05` | Default sleep in seconds between fake-model tokens. The client can override it per response with `--tokendelay`. |
| `STEERABLE_CONVERSATIONS` | `false` | Enables or disables steerable conversations for the server deployment. |
| `FOUNDRY_PROJECT_ENDPOINT` | None | Foundry project endpoint used when a real model is enabled. |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | None | Foundry model deployment name. When unset, the deterministic fake model is used. |

`ResponsesHostServer` advertises the capability in each response as metadata
`{"foundry.agent.steerable_conversation": "true"}` or
`{"foundry.agent.steerable_conversation": "false"}`. The sample client uses
this metadata to keep the normal composer enabled only when an active turn can
be replaced safely.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the
README in the parent directory to run the agent host.

Quick local start (no Azure login or endpoint needed):

```bash
uv sync
uv run python main.py
```

To use a real model, set the project endpoint and deployment name in `.env`,
then start the server with the same command:

```dotenv
FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-4.1-mini"
```

> **Install note.** The resilience API
> (`resilient_background`, `context.is_recovery`, `context.persisted_response`,
> `context.conversation_chain_metadata`, `stream.checkpoint()`) is **not**
> yet on `main` or the PyPI wheels. `requirements.txt` therefore pins both
> `azure-ai-agentserver-core` and `azure-ai-agentserver-responses` to the
> GitHub feature branch `feature/agentserver-durable-agent-demo` (installed
> from source for authenticity). Because the branch reuses the same version
> strings as the published wheels, pip may report them "already satisfied" —
> use `pip install --force-reinstall --no-deps <git-url>` if the recovery
> attributes are missing.

## Interacting with the agent

### Responses protocol — full pipeline round-trip

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
   -d '{"input": "Book a two-night trip to Paris", "background": true, "store": true}'
```

The first response completes with an `mcp_approval_request` before booking. The
Textual client displays the proposed action and arguments; click **Approve** to
resume and execute `book_trip`, or **Deny** to reject it without booking.

The Textual CUI reads the first request and every subsequent turn from one
persistent composer:

```bash
cd client
uv run python client.py \
   --endpoint https://<account>.services.ai.azure.com/api/projects/<project> \
   --agent langchain-azure-resilient-responses-steerable
```

### Streaming

Add `"stream": true` to the body to receive SSE events for every tool
round-trip and the final assistant message.

### Approval, steering, and cancellation

Run the CUI against the steerable deployment and enter the initial request:

```bash
cd client
uv run python client.py \
   --endpoint https://<account>.services.ai.azure.com/api/projects/<project> \
   --agent langchain-azure-resilient-responses-steerable \
   --tokendelay 0.5
```

There is no separate steering mode or queue. While output is active, type the
replacement text into the same composer and press Enter. The replacement
response becomes current immediately and uses the active response as its
parent. Late events remain attached to the superseded transcript turn. Use
**Approve** or **Deny** when `book_trip` reaches the durable approval interrupt.
Use the Cancel button or press `Ctrl+C` once to cancel the current response.
Press `Ctrl+C` again within two seconds to exit; `Ctrl+Q` always exits directly.

For HTTPS Foundry endpoints, the CUI acquires a bearer token with
`DefaultAzureCredential` using the Azure AI scope. Use `--no-auth` for an
unauthenticated host.

### Resilient background streaming + crash recovery

The target scenario combines background execution, resilience, and streaming:

```bash
curl -N -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "crash", "background": true, "stream": true}'
```

This runs the turn as a **resilient background task**. The Responses layer
persists replayable output under `${AGENTSERVER_STATE_ROOT}`. At each native
LangGraph checkpoint, the task records the exact `thread_id` and
`checkpoint_id` needed to resume graph execution.

Before graph work starts, the handler persists admission. It does not store the
resolved thread or parent checkpoint on the current Response. The first valid
LangGraph checkpoint adds the current Response's `langgraph_thread_id` and
`langgraph_checkpoint_id`.

On recovery, a persisted current-response checkpoint resumes LangGraph from
that exact checkpoint. If the current response has not persisted a checkpoint,
the handler starts from the response's original thread or parent checkpoint.
A crash between the LangGraph commit and the Responses checkpoint can repeat
work, providing at-least-once execution across the two checkpoint stores.

To observe recovery locally, use the sample client. Terminal 1:

```bash
AGENTSERVER_STATE_ROOT="$PWD/.agentserver-demo" \
CHECKPOINT_DB="$PWD/demo-checkpoints.sqlite" \
python main.py
```

Terminal 2:

```bash
cd client
uv run python client.py \
   --endpoint https://<account>.services.ai.azure.com/api/projects/<project> \
   --agent langchain-azure-resilient-responses-steerable \
   --tokendelay 0.25
```

You should see the first TODO checked, then `Network connection lost; resuming
automatically...` in the status bar when the process is killed. This means the
CUI is waiting for the same stored response, not creating a new request:

```text
streaming:
... Done planning.
LangGraph checkpointed TODO state:
- [x] 1. Plan the work
- [ ] 2. Research the details
- [ ] 3. Summarize the result

```

Restart Terminal 1 with the same `AGENTSERVER_STATE_ROOT` and `CHECKPOINT_DB`.
The open CUI reconnects automatically through `AsyncOpenAI.responses.retrieve`
with the last received cursor. The recovered output continues from the
checkpointed TODO state and ends with all four TODOs checked:

```text
final responses:
... Done summarizing.
LangGraph checkpointed TODO state:
- [x] 1. Plan the work
- [x] 2. Research the details
- [x] 3. Execute the plan
- [x] 4. Summarize the result
```

> **Local-dev note.** The file-backed stream store guards each stream with a
> `<id>.jsonl.lock` file. A hard kill (`SIGKILL` / force-stop) leaves this lock
> behind, which blocks the recovered attempt with a "lock-file contention"
> error. In a real deployment stream ownership uses lease TTLs, but locally you
> may need to delete the stale `.jsonl.lock` before restarting. A graceful
> shutdown (`Ctrl-C`) releases it automatically.

## Deploying the Agent to Foundry

The sample is its own azd project and deploys directly from this directory.
`deploy.ps1` first builds the repository's current `libs/azure-ai` package into
`vendor/`, so unpublished hosting changes are included without copying the
sample or library source to another project.

On the first deployment, choose an environment, subscription, and region:

```powershell
.\deploy.ps1 `
   -Environment resilient `
   -SubscriptionId "<subscription>" `
   -Location "<region>"
```

Following the official Foundry hosted-agent samples, the `ai-project` service
in `azure.yaml` owns the Foundry project and declares a `gpt-4.1-mini` model
deployment. `deploy.ps1` runs `azd provision` to create or update both before
it deploys the hosted agents, so no project endpoint or model name is supplied
separately.

The azd environment stores the provisioned project outputs. By default, every
subsequent run deploys two agents from the same code: the non-steerable
`langchain-azure-resilient-responses` and the steerable
`langchain-azure-resilient-responses-steerable`:

```powershell
.\deploy.ps1
```

Deploy only one variant when needed:

```powershell
.\deploy.ps1 -Deployment NonSteerable
.\deploy.ps1 -Deployment Steerable
```

Model provisioning is idempotent; subsequent runs update the declared model
only when its configuration changes.
