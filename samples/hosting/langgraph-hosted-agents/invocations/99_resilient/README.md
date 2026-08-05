# Sample 99 — Resilient Invocations + LangGraph checkpointer

> **Work in progress / experimental.** This sample is the sandbox for
> integrating resilient **Invocations protocol** requests from
> `azure-ai-agentserver-invocations` with LangGraph's native checkpointer,
> and for the corresponding changes to
> `langchain_azure_ai.agents.hosting.InvocationsHostServer`.

## What this sample demonstrates

This is a real-model trip-planning agent built as a
[LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` and hosted
over the **Invocations protocol**. Flight and hotel searches run automatically,
but the sensitive `book_trip` tool is blocked by a durable `interrupt()` until
the client sends an explicit approval.

```text
START -> agent -> search tools -> agent -> approval -> book_trip -> agent -> END
```

`DurableInvocationsHostServer` emits the pause as an `approval_required` event.
The Textual client displays the tool arguments and enables **Approve** and
**Deny**.

### Progression (all done)

1. **[done] Baseline** — real-model trip-planning pipeline, running locally.
2. **[done] Enable resilience** — a persistent LangGraph checkpointer
   (`AsyncSqliteSaver`) keeps graph state under a stable `agent_session_id` so
   it survives a process restart.
3. **[done] Recovery-aware hosting** — `DurableInvocationsHostServer` inspects
   the saved LangGraph snapshot and resumes unfinished graph work with input
   `None` when the client retries the invocation.
4. **[done] Session checkpoint continuity** — each request maps the stable
   `agent_session_id` to the same LangGraph thread and latest checkpoint.
5. **[done] Crash-recovery test** — killing the server during the
   `simulate_crash` tool and restarting resumes the pending graph work.
6. **[done] Async CUI** — a Textual client streams, reconnects, and cancels the
   local request.
7. **[done] Human approval** — `book_trip` executes only after `approve`
   resumes the saved LangGraph interrupt.

### Scope

Supports regular and streaming Invocations requests. Recovery is client-driven:
after a disconnect, the client retries with the same `agent_session_id` and the
host resumes any unfinished LangGraph checkpoint.

## Resilience contract

The Invocations protocol does not provide a server-managed background task or
automatic host re-entry. Application code must persist its workflow state, and
the client must retry the request after the host restarts.

### Client responsibilities

- Choose a stable `agent_session_id` and reuse it for every turn, approval, and
   recovery attempt in the conversation.
- After a disconnect, retry the invocation with the same session ID. The host
   resumes unfinished graph work instead of adding the retried message again.
- Treat transport loss as an unknown execution state, not as failure. The
   invocation may have committed graph state before the host stopped.
- Continue conversations linearly. This integration keeps one latest LangGraph
   checkpoint per session; it does not support forking an older turn into a
   second branch.

The sample CUI implements these rules: it keeps the session ID and retries the
same streaming POST after a transport failure until `--reconnect-timeout`
expires.

### Graph and handler responsibilities

- Compile the graph with a durable checkpointer. The checkpointer must survive
   process replacement and be accessible to every host instance that can
   receive the retried invocation.
- Keep all durable workflow state in LangGraph state. Module globals, process
   memory, local caches, temporary files, and active HTTP requests are
   transient and are reconstructed or lost after restart.
- Make graph nodes replay-safe. A crash after an external action but before the
   next LangGraph checkpoint can cause that action to execute again.
- Make external side effects idempotent or deduplicate them with a stable
   operation key stored in graph state. This includes writes, payments, email,
   queue publication, tool calls, and calls to systems that mutate state.
   Recording only "completed" after a side effect is not sufficient: the
   process can crash after the side effect succeeds but before that state is
   checkpointed.
- Keep checkpointed state serializable and compatible across deployments. A
   recovered invocation may load state written by the previous application
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
recover using only its stored session ID.

## How It Works

### Tool-using graph and crash recovery

The graph keeps conversation messages in LangGraph state. The Foundry model
first calls `search_flights` and `search_hotels`, recommends an itinerary, and
then calls `book_trip`. The application pauses that sensitive tool call with a
LangGraph interrupt before it executes.

The `simulate_crash` tool provides a deterministic recovery boundary. On its
first execution it terminates the process. When the client retries after the
host restarts, the host finds the unfinished tool node and resumes the graph
with `invocation_recovery=True`; the tool then reports successful recovery
instead of terminating the new process.

The output is intended to make both layers visible:

- **Invocations retry** — the CUI reports a lost connection and retries the
  same POST with the same `agent_session_id`.
- **LangGraph checkpointing** — the restarted host finds the pending graph node
  in the persistent checkpointer and continues without adding a duplicate user
  message.

### Invocations hosting

`langchain_azure_ai.agents.hosting.InvocationsHostServer` exposes the compiled
graph through the Invocations protocol. This sample subclasses it to detect and
resume unfinished checkpoints:

```python
server = DurableInvocationsHostServer(graph)
await server.run_async(port=int(os.environ.get("PORT", "8088")))
```

The client continues the interactive session with `agent_session_id`. Before
each invocation, the host reads that session's latest LangGraph snapshot. A
pending interrupt accepts only `approve` or `reject`; another pending node is
resumed with input `None`; otherwise the message starts a new turn.

Checkpointed conversations are linear. If the graph completed but the HTTP
response was lost, retrying the request starts a new turn because Invocations
does not provide a response ID or an idempotency key.

The graph is compiled with a persistent checkpointer so state survives a
restart:

```python
async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
    graph = build_graph(checkpointer, model)
    ...
```

`AsyncSqliteSaver` (not the sync `SqliteSaver`) is required because the host
drives the graph with the async API (`ainvoke` / `aget_state`).

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8088` | HTTP port for the agent host. |
| `CHECKPOINT_DB` | `checkpoints.sqlite` (cwd) locally; `$HOME/checkpoints.sqlite` when hosted | SQLite file backing the LangGraph checkpointer. Reuse the same path when restarting the sample. An explicit value always wins. Hosted mode is detected with `AgentConfig.from_env().is_hosted`. |
| `FOUNDRY_PROJECT_ENDPOINT` | None | Required Foundry project endpoint used by the model client. |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | None | Required Foundry model deployment name. |

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the
README in the parent directory to run the agent host.

Set the project endpoint and deployment name in `.env`:

```dotenv
FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-4.1-mini"
```

Authenticate and start the server:

```bash
az login
uv sync
uv run python main.py
```

## Interacting with the agent

### Invocations protocol — full pipeline round-trip

```bash
curl -X POST \
  'http://127.0.0.1:8088/invocations?agent_session_id=trip-demo' \
  -H "Content-Type: application/json" \
  -d '{"message": "Book a two-night trip to Paris"}'
```

The first invocation completes with `status: "approval_required"` before
booking. The Textual client displays the proposed action and arguments; click
**Approve** to resume and execute `book_trip`, or **Deny** to reject it without
booking.

The Textual CUI reads the first request and every subsequent turn from one
persistent composer:

```bash
cd client
uv sync
uv run python client.py --session-id trip-demo
```

### Streaming

Add `"stream": true` to the body to receive SSE token events, followed by an
`approval_required` event when `book_trip` pauses and the final `done` event.

### Approval, recovery, and cancellation

Run the CUI with a stable session ID:

```bash
cd client
uv run python client.py \
  --session-id trip-demo \
  --reconnect-timeout 120
```

Wait for the active invocation to finish before sending another normal message.
Use **Approve** or **Deny** when `book_trip` reaches the durable approval
interrupt. The Cancel button or the first `Ctrl+C` cancels the local HTTP
request. Press `Ctrl+C` again within two seconds to exit; `Ctrl+Q` always exits
directly. `Ctrl+R` toggles the raw SSE event view.

For HTTPS Foundry endpoints, pass the full Invocations URL with `--url` and use
`--auth` to acquire a bearer token from `DefaultAzureCredential` with the Azure
AI scope.

### Invocations streaming + crash recovery

The target scenario combines a streaming request, client retry, and durable
LangGraph state:

```bash
curl -N -X POST \
  'http://127.0.0.1:8088/invocations?agent_session_id=crash-demo' \
  -H "Content-Type: application/json" \
  -d '{"message": "Call simulate_crash, recover, and report the result", "stream": true}'
```

This runs a regular streaming invocation. The Invocations layer does not store
replayable output. The LangGraph checkpointer stores the session's pending graph
state so a later request with the same `agent_session_id` can resume it.

On recovery, `DurableInvocationsHostServer` sees the unfinished snapshot and
resumes LangGraph from that checkpoint with input `None`. A crash between an
external side effect and the next LangGraph checkpoint can repeat work, so
side-effecting tools still need idempotency.

To observe recovery locally, use the sample client. Terminal 1:

```bash
CHECKPOINT_DB="$PWD/demo-checkpoints.sqlite" \
uv run python main.py
```

Terminal 2:

```bash
cd client
uv run python client.py \
  --session-id crash-demo \
  --reconnect-timeout 300
```

Ask the agent to call `simulate_crash`. You should see
`Connection lost. Retrying invocation...` when the process is killed. This
means the CUI is creating a retry with the same session ID:

```text
[Connection lost. Retrying invocation...]
```

Restart Terminal 1 with the same `CHECKPOINT_DB`. The open CUI retries
automatically, and the recovered invocation continues from the pending tool
node and completes the turn.

> **Local-dev note.** The CUI keeps text received before the disconnect, but
> Invocations has no SSE replay cursor. A retried graph node can therefore emit
> duplicate text.

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
it deploys the hosted agent, so no project endpoint or model name is supplied
separately.

The azd environment stores the provisioned project outputs. Every subsequent
run deploys `langchain-azure-resilient-invocations` from the same code:

```powershell
.\deploy.ps1
```

Model provisioning is idempotent; subsequent runs update the declared model
only when its configuration changes.