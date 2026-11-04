# Invocations hosting behavior

## 1. Scope and defaults

This is a high-level summary of the default `InvocationsHostServer` on
[main at `8fabe31`](https://github.com/langchain-ai/langchain-azure/commit/8fabe31),
reviewed on 2026-11-04. Hosting is experimental. The lockfile uses Agent Server
Core `2.3.0b2`, Responses `2.4.0b2`, and Invocations `1.3.0b1`.

[Agent Server Invocations](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-invocations/README.md#protocol-endpoints)
provides HTTP routing and request context. This host defines the text/HITL body
format and connects it to a LangGraph graph or compatible runnable. See
[Responses hosting](responses_protocol.md) for richer input and output.
✓ means supported under the stated conditions; × means unsupported.

| Control | Default | Purpose |
| --- | --- | --- |
| Request `stream` / `background` | `false` | Select SSE delivery / background execution. |
| Server `resilient_background` / `steerable_conversations` | `False` | Enable background recovery / steering. |
| Request `previous_invocation_id` | Unset | Require continuation from the latest invocation. |
| Graph checkpointer | Application-supplied | Save graph state across turns, interrupts, and restarts. |

The Python constructor accepts server options through `ResponsesServerOptions`.
The [CLI](../../langchain_azure_ai/agents/hosting/run.py) supports
`--protocol invocations`; configure these options through Python because
`--option` is only accepted for Responses.

## 2. Calling modes and input/output

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

Enabling either server option puts all requests, including foreground requests,
on the task-backed execution path. It also enables
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
