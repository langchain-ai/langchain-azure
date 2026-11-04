# Invocations hosting behavior

## 1. Interface

`InvocationsHostServer` hosts a LangGraph graph or compatible runnable through
HTTP invocations. Agent Server provides routing and request context; this host
defines the text/HITL body format. Hosting is experimental. See
[Responses hosting](responses_protocol.md) for richer input and output.

### Dependencies

Install `langchain-azure-ai[hosting]`. The
[shared dependency requirements](responses_protocol.md#dependencies) cover
LangChain, LangGraph, and the three Agent Server packages. Deep Agents is
optional and uses the same graph interface; its sample pin is listed there.

### Constructor

| Constructor parameter | Default | Purpose |
| --- | --- | --- |
| `graph` | Required | The graph or runnable to serve. |
| `app` | `None` | Attach to an existing Invocations app, or create one when omitted. |
| `options` | `None` | Use default `ResponsesServerOptions`: `resilient_background=False` and `steerable_conversations=False`. |
| `output_parser` | `None` | Customize final response text extraction; the default uses the last AI message. |
| `applicationinsights_connection_string` / `graceful_shutdown_timeout` | `None` | Configure telemetry / graceful shutdown on the created app. |

When supplying `app`, configure its host-level settings on that app. The graph's
checkpointer is supplied when compiling the graph. Start serving with `run()`
or `run_async()`; `.app` and `.graph` expose the underlying server and runnable.

Subclass hooks cover request parsing (`parse_request()`,
`parse_execution_options()`), graph input (`build_input()`), configuration
(`build_runnable_config()`, `build_task_runnable_config()`), and response text
(`parse_output()`). See the [host interface](../../langchain_azure_ai/agents/hosting/_invoke_host.py)
for details.

The [CLI](../../langchain_azure_ai/agents/hosting/run.py) supports
`--protocol invocations`. Configure constructor parameters through Python;
`--option` is only accepted for Responses.

## 2. Calling modes and input/output

Request defaults are `stream=false` and `background=false`.
✓ means supported under the stated conditions; × means unsupported.

`POST /invocations` accepts a non-empty `message` string, or structured HITL
answers described below. The default graph input uses `messages`; JSON
returns `{"response": "..."}` with the last AI message's text. Custom graph
state needs custom conversion. Rich media, ordinary tool traces, reasoning,
citations, and usage have no default structured output mapping here.

| Calling mode | Supported | Result / conditions |
| --- | :---: | --- |
| Foreground JSON | ✓ | Waits for the response text and optional HITL `output`. |
| Foreground SSE | ✓ | Emits token data, optional HITL `output_item` events, and `done`. |
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
and platform user to an internal LangGraph thread. A checkpointer restores
prior graph state; without one, each request supplies only its current input.
An in-memory saver retains state within one process; restart recovery requires
a persistent saver. Protocol result storage and graph state have separate
lifetimes.

| Conversation feature | Supported | Conditions / behavior |
| --- | :---: | --- |
| Checkpointed multi-turn state | ✓ | Reuse the same session and user context. |
| Transcript history without a checkpointer | × | Reusing a session does not reconstruct prior messages. |
| Linear parent precondition | ✓ | `previous_invocation_id` requires either server option. A mismatch returns HTTP 409. |
| Branch from an older invocation | × | The parent ID checks the latest turn; it does not select an earlier graph checkpoint. |
| Steering | ✓ | `steerable_conversations=True` queues the next input and supersedes active work. A checkpointer is optional for steering itself. |

On the task path, overlapping turns without steering return HTTP 409. Steering
marks the superseded invocation `cancelled` with code `steered`. The direct
path has no wrapper-owned task queue or latest-parent precondition.

## 4. Interrupts and recovery

### Human-in-the-loop

LangGraph `interrupt()` pauses a checkpointed graph. The host returns paired
`function_call` and `mcp_approval_request` items in `output`, or as SSE
`output_item` events. Answer in the next request's `message` list using
`function_call_output` or `mcp_approval_response`; the host converts the answer
to a LangGraph resume command. This works with foreground JSON/SSE and
background polling.

The [HITL decision rules](responses_protocol.md#human-in-the-loop) are shared
with Responses. Plain text while paused re-emits the pending items; structured
answers require a pending interrupt. A completed invocation can leave the graph
paused for a later answer.

### Restart recovery

The supported background recovery setup uses `resilient_background=True`,
`background=true`, and a persistent graph checkpointer. Enabling resilience
without a checkpointer is rejected at host construction. Agent Server task
state and invocation storage must also survive the restart.

The host resumes from saved graph progress, reuses a saved final result when
available, or processes the original input again when no progress was saved.
External effects can repeat across a recovery boundary. Invocation state is
file-backed locally and Foundry-backed when hosted. Terminal results remain
retrievable for 30 days, separately from short-lived replay events and graph
checkpoint retention.

Cancellation can pass through `cancelling` before becoming `cancelled`.
Foreground SSE has no public reconnect contract when tasks are enabled.
