# Invocations hosting behavior

## 1. Interface

[`InvocationsHostServer`](https://reference.langchain.com/python/langchain-azure-ai/agents/hosting/_invoke_host/InvocationsHostServer) hosts a [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) graph or
compatible runnable through HTTP invocations. The [Agent Server SDK](https://learn.microsoft.com/en-us/python/api/azure-ai-agentserver-core/azure.ai.agentserver.core?view=azure-python) provides
routing and request context; this host defines the text and
[human-in-the-loop (HITL)](https://docs.langchain.com/oss/python/langgraph/interrupts) request formats. Hosting is experimental. See
[Responses hosting](responses_protocol.md) for richer input and output.

### Dependencies

Install `langchain-azure-ai[hosting]>=1.3.0`. The
[shared dependency requirements](responses_protocol.md#dependencies) cover
[LangChain](https://docs.langchain.com/oss/python/langchain/overview), LangGraph, and the three Agent Server packages. [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) is
optional and uses the same graph interface; its sample pin is listed there.

### Constructor

| Constructor parameter | Default | Purpose |
| --- | --- | --- |
| `graph` | Required | The graph or runnable to serve. |
| `app` | `None` | Attach to an existing Invocations app, or create one when omitted. |
| `options` | `None` | Use default [`ResponsesServerOptions`](https://learn.microsoft.com/en-us/python/api/azure-ai-agentserver-responses/azure.ai.agentserver.responses.responsesserveroptions?view=azure-python): `resilient_background=False` and `steerable_conversations=False`. |
| `output_parser` | `None` | Customize final response text extraction; the default uses the last AI message. |
| `applicationinsights_connection_string` / `graceful_shutdown_timeout` | `None` | Configure telemetry / graceful shutdown on the created app. |

> [!NOTE]
> When supplying `app`, configure its host-level settings on that app. Supply
> the graph's [checkpointer](https://docs.langchain.com/oss/python/langgraph/checkpointers) when compiling it.

Start serving with `run()` or `run_async()`; `.app` and `.graph` expose the
underlying server and runnable.

> [!TIP]
> The [CLI](../../langchain_azure_ai/agents/hosting/run.py) supports
> `--protocol invocations`. Configure constructor parameters through Python;
> `--option` is only accepted for Responses.

## 2. Calling modes and input/output

Request defaults are `stream=false` and `background=false`.
✓ means supported under the stated conditions; × means unsupported.

`POST /invocations` accepts a non-empty `message` string, or structured HITL
answers described below. The default graph input uses `messages`; JSON
returns `{"response": "..."}` with the last AI message's text. Other graph
state has no default mapping. Rich media, ordinary tool traces, reasoning,
citations, and usage have no default structured output mapping here.

| Calling mode | Supported | Result / conditions |
| --- | :---: | --- |
| Foreground JSON | ✓ | Waits for the response text and optional HITL `output`. |
| Foreground [SSE](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses) | ✓ | Emits token data, optional HITL `output_item` events, and `done`. |
| Background JSON | ✓ | Requires resilience and a checkpointer; returns HTTP 202 with an invocation ID for polling. |
| Background SSE | × | `background=true` with `stream=true` returns HTTP 400. |
| SSE reconnect by cursor | × | There is no public stream-reconnect endpoint. |

[Agent Server Invocations](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-invocations/README.md#protocol-endpoints)
provides the HTTP endpoints. Enabling either server option puts all requests,
including foreground requests, on the task-backed execution path. It also enables
`GET /invocations/{id}` and `POST /invocations/{id}/cancel`. With both options
off, requests execute directly. Background requests require
`resilient_background=True` even when steering is enabled.

## 3. Conversation and state

Reuse `agent_session_id` for later turns. The host maps the resolved session
and platform user to an internal LangGraph [thread](https://docs.langchain.com/oss/python/langgraph/persistence#threads). A checkpointer restores
prior graph state; without one, each request supplies only its current input.
An in-memory saver retains state within one process; restart recovery requires
a persistent saver. Protocol result storage and graph state have separate
lifetimes.

The same [graph storage choices](responses_protocol.md#storage-choices) apply
here: [`FoundryCheckpointSaver`](https://reference.langchain.com/python/langchain-azure-ai/agents/hosting/_foundry_checkpoint_saver/FoundryCheckpointSaver) for the
[durable state store](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/agent-state-store), async [`CosmosDBSaver`](https://reference.langchain.com/python/langchain-azure-cosmosdb/aio/_langgraph_checkpoint_store/CosmosDBSaver)
for [Azure Cosmos DB for NoSQL](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/overview) checkpoints.

| Conversation feature | Supported | Conditions / behavior |
| --- | :---: | --- |
| Checkpointed multi-turn state | ✓ | Reuse the same session and user context. |
| Transcript history without a checkpointer | × | Reusing a session does not reconstruct prior messages. |
| Linear parent precondition | ✓ | `previous_invocation_id` requires either server option. A mismatch returns HTTP 409. |

### Steering

[Steering](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-responses/docs/resilient-responses-developer-guide.md) uses `steerable_conversations=True` to queue the next input
and supersede active work. It does not require a checkpointer by itself.
Overlapping task-backed turns without steering return HTTP 409. Steering marks
the superseded invocation `cancelled` with code `steered`. The direct execution
path has no wrapper-owned task queue or latest-parent precondition.

### Branching

Branching from an older invocation is unsupported. `previous_invocation_id`
checks the latest turn; it does not select an earlier graph checkpoint.

## 4. Interrupts and recovery

### Human-in-the-loop

LangGraph [`interrupt()`](https://docs.langchain.com/oss/python/langgraph/interrupts) pauses a checkpointed graph. The host returns paired
`function_call` and `mcp_approval_request` items in `output`, or as SSE
`output_item` events. Answer in the next request's `message` list using
`function_call_output` or `mcp_approval_response`; the host converts the answer
to a LangGraph resume command. This works with foreground JSON/SSE and
background polling.

> [!NOTE]
> The [two HITL reply paths and event examples](responses_protocol.md#human-in-the-loop)
> also apply here. Put the reply items in `message` and reuse `agent_session_id`;
> Responses uses `input`.

Plain text while paused re-emits the pending items; structured answers require
a pending interrupt. A completed invocation can leave the graph paused for a
later answer.

### Restart recovery

The supported background recovery setup uses `resilient_background=True`,
`background=true`, and a persistent graph checkpointer. Enabling resilience
without a checkpointer is rejected at host construction. Agent Server task
state and invocation storage must also survive the restart.

The host resumes from saved graph progress, reuses a saved final result when
available, or processes the original input again when no progress was saved.

> [!WARNING]
> External effects can repeat across a recovery boundary.

Invocation state is backed by local files or, when hosted, the durable state store.
Terminal results remain retrievable for 30 days, separately from replay events
and graph checkpoint retention.

Cancellation can pass through `cancelling` before becoming `cancelled`.
Foreground SSE has no public reconnect contract when tasks are enabled.
