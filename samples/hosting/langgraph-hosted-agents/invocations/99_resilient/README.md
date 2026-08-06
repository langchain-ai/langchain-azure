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

`InvocationsHostServer` exposes pending interrupts as the same paired
`function_call` and `mcp_approval_request` items as `ResponsesHostServer`, and
accepts the matching structured response items to resume the graph.

### Progression (all done)

1. **[done] Baseline** — real-model trip-planning pipeline, running locally.
2. **[done] Enable resilience** —
   `ResponsesServerOptions(resilient_background=True)` enables durable
   invocation tasks, while `AsyncSqliteSaver` persists LangGraph state.
3. **[done] Recovery-aware hosting** — `InvocationsHostServer` is re-entered
   after a process restart and resumes from the exact persisted LangGraph
   checkpoint with input `None`.
4. **[done] Session checkpoint continuity** — each request maps the stable
   `agent_session_id` to the same LangGraph thread and supports a linear
   `previous_invocation_id` precondition.
5. **[done] Crash-recovery test** — killing the server during the
   `simulate_crash` tool and restarting resumes the pending graph work.
6. **[done] Retrieval and cancellation** — background work can be polled with
   `GET /invocations/{invocation_id}` and cancelled with
   `POST /invocations/{invocation_id}/cancel`.
7. **[done] Human approval** — `book_trip` executes only after a structured
   HITL response resumes the saved LangGraph interrupt.

### Scope

Supports foreground, streaming, and resilient background Invocations requests.
Background streaming isn't supported; resilient callers receive `202` and poll
the invocation resource instead.

## Resilience contract

With `resilient_background=True`, `InvocationsHostServer` runs background work
as a durable task. After a host restart, Agent Server re-enters the same task
and the host resumes the paired LangGraph checkpoint.

### Client responsibilities

- Choose a stable `agent_session_id` and reuse it for every turn and approval
   in the conversation.
- Save the invocation ID returned by the initial `202` response. After a
   disconnect or host restart, poll that same invocation; don't submit the
   create request again.
- Treat transport loss as an unknown execution state, not as failure. The
   invocation may have committed graph state before the host stopped.
- Continue conversations linearly. Send the latest invocation ID as
   `previous_invocation_id` when starting the next turn.

### Graph and handler responsibilities

- Compile the graph with a durable checkpointer. `InvocationsHostServer`
   rejects `resilient_background=True` when the graph has no checkpointer. The
   checkpointer must survive process replacement and be accessible to every
   host instance that can recover the invocation.
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
recover using only its stored invocation ID.

## How It Works

### Tool-using graph and crash recovery

The graph keeps conversation messages in LangGraph state. The Foundry model
first calls `search_flights` and `search_hotels`, recommends an itinerary, and
then calls `book_trip`. The application pauses that sensitive tool call with a
LangGraph interrupt before it executes.

The `simulate_crash` tool provides a deterministic recovery boundary. On its
first execution it terminates the process. Agent Server re-enters the durable
task after restart, and `InvocationsHostServer` resumes the saved checkpoint.
The tool observes `invocation_context.entry_mode == "recovered"` and reports
successful recovery instead of terminating the new process.

The output is intended to make both layers visible:

- **Invocations recovery** — the caller keeps polling the same durable
   invocation ID while Agent Server re-enters interrupted work.
- **LangGraph checkpointing** — the restarted host finds the pending graph node
  in the persistent checkpointer and continues without adding a duplicate user
  message.

### Invocations hosting

`langchain_azure_ai.agents.hosting.InvocationsHostServer` exposes the compiled
graph through the Invocations protocol. Resilience is opt-in:

```python
options = ResponsesServerOptions(resilient_background=True)
server = InvocationsHostServer(graph, options=options)
await server.run_async(port=int(os.environ.get("PORT", "8088")))
```

The host creates the durable task, records exact checkpoint references,
translates LangGraph interrupts to structured HITL items, exposes retrieval and
cancel routes, and resumes interrupted work after restart.

The graph is compiled with a persistent checkpointer so state survives a
restart:

```python
async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
    graph = build_graph(checkpointer, model)
    ...
```

`AsyncSqliteSaver` (not the sync `SqliteSaver`) is required because the host
drives the graph with the async API (`astream` / `aget_state`).

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8088` | HTTP port for the agent host. |
| `CHECKPOINT_DB` | `checkpoints.sqlite` (cwd) locally; `$HOME/checkpoints.sqlite` when hosted | SQLite file backing the LangGraph checkpointer. Reuse the same path when restarting the sample. An explicit value always wins. Hosted mode is detected with `AgentConfig.from_env().is_hosted`. |
| `AGENTSERVER_STATE_ROOT` | `~/.agentserver` | Root of the local durable task and invocation event stores. Reuse it across restarts. |
| `STEERABLE_CONVERSATIONS` | `false` | Allows a newer turn to supersede active work in the same session. |
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
   -d '{"message": "Book a two-night trip to Paris", "background": true}'
```

The server returns `202` with an invocation envelope:

```json
{
   "id": "<invocation-id>",
   "status": "queued",
   "agent_session_id": "trip-demo"
}
```

Poll until the invocation reaches a terminal status:

```bash
curl http://127.0.0.1:8088/invocations/<invocation-id>
```

When the graph pauses before `book_trip`, the terminal envelope contains an
`output` array with paired `function_call` and `mcp_approval_request` items.
Save the approval request's `id`, then start the approval as the next turn in
the same session:

```bash
curl -X POST \
   'http://127.0.0.1:8088/invocations?agent_session_id=trip-demo' \
   -H "Content-Type: application/json" \
   -d '{
      "message": [{
         "type": "mcp_approval_response",
         "approval_request_id": "<approval-request-id>",
         "approve": true
      }],
      "background": true,
      "previous_invocation_id": "<invocation-id>"
   }'
```

To reject the tool call without booking, resume the paired `function_call`
with a false value, using its emitted `call_id`:

```json
{
   "message": [{
      "type": "function_call_output",
      "call_id": "<interrupt-call-id>",
      "output": "{\"resume\": false}"
   }]
}
```

An MCP response with `"approve": false` rejects the invocation itself and
leaves the graph interrupt pending. Foreground requests can use
`"stream": true`; pending items arrive as `output_item` SSE events.
`background` and `stream` can't both be true.

### Cancellation

```bash
curl -X POST http://127.0.0.1:8088/invocations/<invocation-id>/cancel
```

### Resilient background crash recovery

Start a background invocation that asks the agent to crash:

```bash
curl -X POST \
  'http://127.0.0.1:8088/invocations?agent_session_id=crash-demo' \
  -H "Content-Type: application/json" \
   -H "x-agent-invocation-id: crash-demo-1" \
   -d '{
      "message": "Call simulate_crash, recover, and report the result",
      "background": true
   }'
```

The POST returns before graph execution finishes. Keep polling the known ID:

```bash
curl http://127.0.0.1:8088/invocations/crash-demo-1
```

After the process terminates, restart it from the same directory with the same
`AGENTSERVER_STATE_ROOT`. Agent Server re-enters `crash-demo-1`, and the host
resumes from the paired LangGraph checkpoint. Continue polling the same ID; do
not send the original POST again. Side-effecting tools still need idempotency
because a crash can occur between an external effect and checkpoint commit.

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