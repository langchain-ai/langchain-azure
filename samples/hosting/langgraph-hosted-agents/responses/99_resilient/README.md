# Sample 99 — Resilient background Responses + LangGraph checkpointer

> **Work in progress / experimental.** This sample is the sandbox for
> integrating the **resilient background responses** feature of
> `azure-ai-agentserver-responses` with LangGraph's native checkpointer,
> and for the corresponding changes to
> `langchain_azure_ai.agents.hosting.ResponsesHostServer`.

## What this sample demonstrates

Currently this is a **deterministic (LLM-free)** TODO-checklist workflow built
as a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph`,
hosted as the **Responses protocol**. The graph runs
`plan -> [research] -> [execute] -> summarize`; planning decides whether the
bracketed phases are needed. Each executed phase marks one TODO item complete
and emits the full checklist, so the output visibly shows what state was
persisted at each checkpoint. **No Azure credentials or model deployment are
required for local runs.**

### Progression (all done)

1. **[done] Baseline** — deterministic numeric pipeline, running locally.
2. **[done] Enable resilience** — `ResponsesServerOptions(resilient_background=True)`
   plus a persistent LangGraph checkpointer (`AsyncSqliteSaver`) so graph
   state survives a process restart.
3. **[done] Recovery-aware hosting** — `ResponsesHostServer.handle_create`
   branches on `context.is_recovery`, resolves the LangGraph `thread_id`
   from `context.conversation_chain_id`, resumes from the native LangGraph
   checkpoint (running only the remaining nodes), seeds the reconnecting
   stream from `context.persisted_response`, and emits a
   `response.in_progress` reset instead of re-running the whole turn.
4. **[done] Per-phase checkpointing** — the streaming converter
   `yield`s `stream.checkpoint()` at each node (phase) boundary and records
   the LangGraph `thread_id`, `last_node`, and a monotonic `phase` watermark
   on the response's `internal_metadata` (persisted with each checkpoint,
   stripped from client payloads).
5. **[done] Crash-recovery test** — killing the server mid-turn and
   restarting resumes the response to completion with an identical output.

### Scope

Supports `background=true` + `resilient_background=true` + `stream=true`.
Steerable conversations can be enabled per deployment with
`STEERABLE_CONVERSATIONS=true`.

## How It Works

### Deterministic TODO checklist (no LLM)

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

The output is intended to make both layers visible:

- **Responses checkpointing** — the stream shows text emitted before the crash,
  then the client prints `[retrying...]` and reconnects/polls the same response.
- **LangGraph checkpointing** — the TODO list after recovery already contains
  `- [x] 1. Plan the work`, proving graph state survived the process crash.

### Responses hosting

`langchain_azure_ai.agents.hosting.ResponsesHostServer` exposes the
compiled graph through the Responses protocol. Resilience is opt-in:

```python
options = ResponsesServerOptions(resilient_background=True)
await ResponsesHostServer(graph, options=options).run_async(port=port)
```

When a turn uses `previous_response_id`, the host reads the immediate parent
Response once. Its internal metadata contains the stable LangGraph thread ID
and exact checkpoint ID needed to continue or fork that parent.

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
| `TOKEN_DELAY_SECONDS` | `0.05` | Default sleep in seconds between fake-model tokens. The client can override it per response with `--token-delay`. |
| `STEERABLE_CONVERSATIONS` | `false` | Enables or disables steerable conversations for the server deployment. |

`ResponsesHostServer` advertises the capability in each response as metadata
`{"foundry.agent.steerable_conversation": "true"}` or
`{"foundry.agent.steerable_conversation": "false"}`. The sample client uses
this metadata to show `s` only when an active turn can actually be steered
rather than forked.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the
README in the parent directory to run the agent host.

Quick local start (no Azure login or endpoint needed):

```bash
pip install -r requirements.txt
python main.py
```

> **Install note.** The resilience API
> (`resilient_background`, `context.is_recovery`, `context.persisted_response`,
> `context.conversation_chain_id/_metadata`, `stream.checkpoint()`) is **not**
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
  -d '{"input": "go"}'
```

The `output` array contains one assistant `message` per graph phase. Each
message includes the current TODO checklist.

The sample client reads the first request and all subsequent turns from its
interactive console:

```bash
python client.py --background --stream
```

### Streaming

Add `"stream": true` to the body to receive SSE events for every tool
round-trip and the final assistant message.

### Interactive steering and cancellation

Run the client with background streaming against the steerable deployment,
then enter the initial request at the console prompt:

```bash
python client.py --background --stream --token-delay 0.5
```

After the response starts, the client prints `type s for steer, type c for
cancel`. Enter `s`, then the replacement input, to queue a new turn behind the
active response. The superseded response completes with its checkpointed
partial output; the client then follows the queued response through the same
interactive loop, so it can be steered or cancelled again. The sample model
checks the handler cancellation event between tokens and while waiting for the
next token delay, so steering does not wait for the current graph node to end.

### Resilient background streaming + crash recovery

The target scenario combines background execution, resilience, and streaming:

```bash
curl -N -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "crash", "background": true, "stream": true}'
```

This runs the turn as a **resilient background task**. At each phase boundary
the host persists a checkpoint snapshot (the response's `output` so far plus
the `langgraph_thread_id` / `last_node` / `phase` watermark on
`internal_metadata`) under `${AGENTSERVER_STATE_ROOT}`.

To observe recovery locally, use the sample client. Terminal 1:

```bash
AGENTSERVER_STATE_ROOT="$PWD/.agentserver-demo" \
CHECKPOINT_DB="$PWD/demo-checkpoints.sqlite" \
python main.py
```

Terminal 2:

```bash
python client.py --background --stream --token-delay 0.25
```

You should see the first TODO checked, then `[retrying...]` when the process is
killed:

```text
streaming:
... Done planning.
LangGraph checkpointed TODO state:
- [x] 1. Plan the work
- [ ] 2. Research the details
- [ ] 3. Summarize the result

[retrying...]
```

Restart Terminal 1 with the same `AGENTSERVER_STATE_ROOT` and `CHECKPOINT_DB`,
then poll the response id printed by the client:

```bash
python client.py --poll <response_id>
```

The recovered output should continue from the checkpointed TODO state and end
with all three TODOs checked:

```text
final responses:
... Done summarizing.
LangGraph checkpointed TODO state:
- [x] 1. Plan the work
- [x] 2. Research the details
- [x] 3. Summarize the result
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

Bind an existing Foundry project on the first deployment:

```powershell
.\deploy.ps1 `
   -Environment ai-test `
   -ProjectEndpoint "https://<account>.services.ai.azure.com/api/projects/<project>" `
   -ProjectId "/subscriptions/<subscription>/resourceGroups/<group>/providers/Microsoft.CognitiveServices/accounts/<account>/projects/<project>" `
   -SubscriptionId "<subscription>" `
   -TenantId "<tenant>" `
   -Location "<project-region>"
```

The script stores those values in the local azd environment. By default, every
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

Do not run `azd provision` when binding an existing Foundry project.
