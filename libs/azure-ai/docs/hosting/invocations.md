# Invocations hosting overview

The hosting integration in `langchain-azure-ai` is built on the
[Agent Server SDK](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-invocations/README.md).
It implements the SDK's handler interface so
[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) agents can
run as [hosted agents in Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/hosted-agents).

[`InvocationsHostServer`](../../langchain_azure_ai/agents/hosting/_invoke_host.py) maps a graph or compatible runnable to HTTP invocations.
It converts text and [human-in-the-loop (HITL)](https://docs.langchain.com/oss/python/langgraph/interrupts)
requests into graph input and resume commands, converts graph output into JSON
or SSE, and connects graph state to session and recovery context. The SDK provides
protocol routing and the server lifecycle. See [Responses hosting](responses.md)
for richer input and output.

## 1. Interface

### Dependencies

Install `langchain-azure-ai[hosting]`. The
[shared dependency requirements](responses.md#dependencies) cover
[LangChain](https://docs.langchain.com/oss/python/langchain/overview), LangGraph,
and the Agent Server SDK. [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
is optional and uses the same graph interface.

### Constructor

Import the host from `langchain_azure_ai.agents.hosting`. The class is generic
over the runnable's input and output types:
`InvocationsHostServer[GraphInputT, GraphOutputT]`. Pass `graph` as a positional
or keyword argument; all other constructor parameters are keyword-only.

| Constructor parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `graph` | `Runnable[GraphInputT, GraphOutputT]` | Required | The graph or runnable to serve. |
| `options` | `ResponsesServerOptions \| None` | `None` | Use default [`ResponsesServerOptions`](https://learn.microsoft.com/en-us/python/api/azure-ai-agentserver-responses/azure.ai.agentserver.responses.responsesserveroptions?view=azure-python): `resilient_background=False` and `steerable_conversations=False`. These two options configure invocation execution; when composed with `ResponsesAgentServerHost`, the same options configure that protocol host. |
| `output_parser` | `Callable[[GraphOutputT], str] \| None` | `None` | Synchronously extract final non-streaming response text; the default uses the last AI message. This does not change SSE token output. |
| `applicationinsights_connection_string` | `str \| None` | SDK default | Forward telemetry configuration to the SDK host. |
| `graceful_shutdown_timeout` | `int \| None` | SDK default | Forward graceful shutdown configuration to the SDK host. |
| `**kwargs` | `Any` | SDK defaults | Forward additional keyword arguments unchanged to `InvocationAgentServerHost`. |

SDK host keywords include `configure_observability`, `log_level`, `access_log`,
`access_log_format`, `routes`, and `openapi_spec`. Omitted values use SDK
defaults; explicit `None` values are preserved. For example,
`configure_observability=None` disables automatic SDK observability setup.
See the [SDK constructor reference](https://learn.microsoft.com/en-us/python/api/azure-ai-agentserver-invocations/azure.ai.agentserver.invocations.invocationagentserverhost?view=azure-python)
for the options supported by your installed version.

The host, `InvocationAgentServerHost`, and `ResponsesServerOptions` are available
from `langchain_azure_ai.agents.hosting`. Import `Runnable` from
`langchain_core.runnables` and `Callable` from `collections.abc`. The generic
types describe the runnable; request parsing and output conversion still use
the defaults below unless customized.

> [!NOTE]
> Supply the graph's [checkpointer](https://docs.langchain.com/oss/python/langgraph/checkpointers) when compiling it.

### Lifecycle and accessors

| Member | Result / behavior |
| --- | --- |
| `run(host: str = "0.0.0.0", port: int \| None = None) -> None` | Start serving synchronously. |
| `async run_async(host: str = "0.0.0.0", port: int \| None = None) -> None` | Start serving in an async application; await this method. |
| `.graph: Runnable[GraphInputT, GraphOutputT]` | Read-only access to the hosted runnable. |

Both start methods delegate to the Agent Server SDK. With `port=None`, the SDK
uses the `PORT` environment variable or port 8088.

The server extends `InvocationAgentServerHost` and is itself an ASGI application.
Pass it directly to ASGI tools, for example `TestClient(server)`, and configure
routes or middleware through `server.add_route(...)` and `server.add_middleware(...)`.

For migration from the earlier wrapper API, replace `server.app` with `server`.
The `app=` constructor parameter has been removed. Configure or subclass the
server itself instead of injecting an existing SDK host.

> [!TIP]
> The [CLI](../../langchain_azure_ai/agents/hosting/run.py) supports
> `--protocol invocations`. Configure constructor parameters through Python;
> `--option` is only accepted for Responses.

### Overridable methods

Override these methods to change request parsing, runnable configuration, or
input/output conversion. The signatures below omit `self`; only the two parsing
methods are asynchronous.

| Method signature | Responsibility |
| --- | --- |
| `async parse_request(request: Request) -> tuple[str \| list[dict[str, Any]], bool]` | Return `(message, stream)`. The default message is text or a list of HITL reply items. |
| `async parse_execution_options(request: Request, *, stream: bool) -> tuple[bool, str \| None]` | Return `(background, previous_invocation_id)`. Override this alongside `parse_request` if a custom request layout moves these controls. Parsing methods raise `ValueError` for invalid input, which becomes HTTP 400. |
| `build_runnable_config(request: Request) -> RunnableConfig` | Configure direct execution. The default derives a user-scoped graph thread from the public session. |
| `build_task_runnable_config(session_id: str, context: TaskContext[dict[str, Any]]) -> RunnableConfig` | Configure task execution, including foreground requests when either server option is enabled. The default uses `context.task_id` for the graph thread and supplies `invocation_context` and `invocation_cancellation_signal`. |
| `build_input(message: str) -> GraphInputT` | Convert text to runnable input. The default creates a `messages` dictionary. Structured HITL replies are handled separately by the host. |
| `parse_output(output: GraphOutputT) -> str` | Convert a non-streaming graph result to the `response` string, using `output_parser` when supplied. SSE tokens use the graph stream instead. |

Import `Request` from `starlette.requests`, `RunnableConfig` from
`langchain_core.runnables`, `TaskContext` from `azure.ai.agentserver.core.tasks`,
and `Any` from `typing`. `GraphInputT` and `GraphOutputT` stand for the input and
output types chosen for the host.

For config changes, call the corresponding method on `super()` and preserve its
thread and task context entries. If customization must work in both execution
paths, override both config methods. The constructor validates the graph schema
before any input conversion runs.

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

[Agent Server SDK](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-invocations/README.md#protocol-endpoints)
provides the HTTP endpoints. Enabling either server option puts all requests,
including foreground requests, on the task-backed execution path. It also enables
`GET /invocations/{id}` and `POST /invocations/{id}/cancel`. With both options
off, requests execute directly. Background requests require
`resilient_background=True` even when steering is enabled.

### HTTP request and foreground result

For example, `POST /invocations` with `Content-Type: application/json`:

```json
{"message": "Hello!", "stream": false, "background": false}
```

`message` is required. `stream` and `background` must be booleans. The optional
`previous_invocation_id` must be a non-empty string and checks the latest turn
when task execution is enabled.

A successful foreground request returns HTTP 200 in both direct and task
execution:

```json
{"response": "Hello! How can I help?"}
```

Pending interrupts add an `output` array of HITL items. Request validation
errors return HTTP 400 with `{"error": "..."}`; conversation conflicts return
HTTP 409 with the same error shape.

### Background submission, polling, and cancellation

Submit `{"message": "Hello!", "background": true}` to receive HTTP 202:

```json
{
  "id": "<invocation-id>",
  "status": "in_progress",
  "agent_session_id": "<session-id>"
}
```

The initial status is `queued` or `in_progress`. The `id` also appears in the
`x-agent-invocation-id` response header. Poll `GET /invocations/{id}` for the
current state. A completed result looks like this:

```json
{
  "id": "<invocation-id>",
  "status": "completed",
  "agent_session_id": "<session-id>",
  "response": "Hello! How can I help?"
}
```

Polling and cancellation return `id`, `status`, and `agent_session_id`, with
`response`, `output`, or `error` when available. `output` contains HITL items;
`error` is an object with `code` and `message`. Status values are `queued`,
`in_progress`, `cancelling`, `completed`, `failed`, and `cancelled`.

`POST /invocations/{id}/cancel` returns the current invocation envelope. Active
work can pass through `cancelling`; a terminal invocation returns its existing
result. Unknown IDs return HTTP 404. Cancellation returns HTTP 409 when the
invocation exists but has no active task in the current process.

### SSE format

With `stream=true`, a foreground request returns `text/event-stream`. Token
events have no event name. A successful stream ends with `done`:

```text
data: {"token": "Hello"}

data: {"token": "!"}

event: done
data: {}

```

Pending interrupts produce `event: output_item` before `done`. Its `data` is the
HITL item itself, without a Responses `response.output_item.done` wrapper.
On execution failure, the stream ends with `event: error` instead of `done`.
Direct execution uses `data: {"error": "..."}`; task execution uses an error
object with `code` and `message`.

## 3. Conversation and state

Reuse the `x-agent-session-id` response header as the next request's
`agent_session_id` query parameter, for example
`POST /invocations?agent_session_id=<session-id>`. The host maps the resolved session
and platform user to an internal LangGraph [thread](https://docs.langchain.com/oss/python/langgraph/persistence#threads). A checkpointer restores
prior graph state; without one, each request supplies only its current input.
An in-memory saver retains state within one process; restart recovery requires
a persistent saver. Protocol result storage and graph state have separate
lifetimes.

The same [graph storage choices](responses.md#storage-choices) apply
here: [`FoundryCheckpointSaver`](../../langchain_azure_ai/agents/hosting/_checkpoint/_foundry_checkpoint_saver.py) for the
[durable state store](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/agent-state-store), async [`CosmosDBSaver`](../../../azure-cosmosdb/src/langchain_azure_cosmosdb/aio/_langgraph_checkpoint_store.py)
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
> Use the [same two HITL reply item types and routing IDs](responses.md#human-in-the-loop).
> Put reply items in `message` and reuse `agent_session_id`; Responses uses
> `input`. The SSE event names and framing differ as described above.

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
