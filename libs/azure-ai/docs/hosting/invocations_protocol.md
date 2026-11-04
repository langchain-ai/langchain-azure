# Invocations protocol: current behavior

This document describes `InvocationsHostServer` on `main` at
[`8fabe31`](https://github.com/langchain-ai/langchain-azure/commit/8fabe31),
reviewed on 2026-11-04. It covers the default host and converters. Subclasses
can change these behaviors. The hosting API is experimental.

The checked-in lockfile uses Agent Server Core `2.3.0b2`, Responses `2.4.0b2`,
and Invocations `1.3.0b1`. See also [Responses protocol](responses_protocol.md).

**Legend:** ✓ supported under the conditions in Notes; × unsupported.
`resilient_background` and `steerable_conversations` are server options, both
false by default. `background` and `stream` are request fields, both false
by default. The Python constructor accepts these options through
`ResponsesServerOptions`, despite the protocol name.

## Features

| Feature | Supported | Notes |
| --- | :---: | --- |
| Foreground JSON request | ✓ | `POST /invocations` with a non-empty `message` string. Returns `{"response": "..."}`; the default parser selects the last AI message's text. |
| Foreground SSE | ✓ | `stream=true` emits `data: {"token": "..."}`, optional `event: output_item` HITL items, and `event: done`. This is a different event schema from Responses. |
| Background execution | ✓ | `background=true` returns HTTP 202 with `id`, `status`, and `agent_session_id`. Requires `resilient_background=True` and a graph checkpointer. |
| Background and streaming in the same request | × | Returns HTTP 400. |
| Retrieve an invocation | ✓ | `GET /invocations/{id}` is wired when either server option is enabled. Returns the latest status envelope, including final response/output when available. |
| Cancel an invocation | ✓ | `POST /invocations/{id}/cancel`, with either option enabled. Active work can report `cancelling`; queued work can be cancelled. A terminal invocation returns its existing terminal envelope. |
| Resume SSE using `starting_after` | × | The wrapper exposes polling, not a public stream-reconnect or cursor endpoint. |
| Multi-turn graph state | ✓ | Reuse `agent_session_id` through the query parameter or `x-agent-session-id` header, with a checkpointer. The public session ID is returned in headers; an internal user/session hash is used for graph state. |
| Transcript replay without a checkpointer | × | Each string becomes current input only. Reusing the session ID does not reconstruct message history. |
| Enforce the latest invocation as parent | ✓ | Request `previous_invocation_id`, with either option enabled. This is a linear-chain precondition; a mismatch returns HTTP 409. It is not checkpoint selection. |
| Branch from an older invocation | × | No equivalent of `enable_response_branching`. An old `previous_invocation_id` fails the latest-turn precondition. |
| Steer an active session | ✓ | `steerable_conversations=True` queues the next turn and supersedes active work. A checkpointer is optional for steering itself; it is needed to preserve graph state. |
| HITL pause/resume | ✓ | A checkpointed graph emits paired interrupt items in `output` or SSE `output_item` events. Resume items are sent as the next request's `message` list. |
| Ordinary tool calls/results as structured wire output | × | Tools run inside the graph; the default wire output is response text plus HITL items. It does not expose a Responses tool trace. |
| Native image/file input, citations, reasoning summaries, or usage fields | × | The default contract accepts text or HITL response items and emits text/HITL. Responses has richer converters for these features. |
| Custom runnable/output shape | ✓ | `output_parser` changes final text extraction. Custom request/state shapes require overriding the input/schema hooks; `output_parser` does not replace SSE token extraction. A graph schema exposing a builder must declare `messages` by default. |
| External host app | ✓ | Pass `app=...` to attach handlers. Configure the underlying app's host-level settings yourself. |
| CLI startup from `langgraph.json` | ✓ | `python -m langchain_azure_ai.agents.hosting.run --protocol invocations`. The CLI rejects `--option`; configure resilience and steering through Python. |

## Server-option combinations

Enabling either option changes **all** invocation requests to the task-backed
path, including foreground requests. With both options off, requests run
directly through the graph. A checkpointer is supplied when compiling the
graph; the host does not add one.

| `resilient_background` | `steerable_conversations` | Checkpointer | Host starts | Execution / notes |
| :---: | :---: | :---: | :---: | --- |
| × | × | × | ✓ | Direct foreground execution; no cross-turn graph memory. |
| × | × | ✓ | ✓ | Direct foreground execution with checkpointed session state. |
| × | ✓ | × | ✓ | Task-backed foreground execution, steering, retrieval, and cancellation; no checkpointed graph memory. Replay events are in memory for a standalone host. |
| × | ✓ | ✓ | ✓ | Same task path with graph state and checkpoint references. Request `background=true` is still rejected. |
| ✓ | × | × | × | Constructor raises `ValueError`. |
| ✓ | × | ✓ | ✓ | Task-backed foreground/background execution; file-backed event replay. Overlapping active turns are rejected instead of steered. |
| ✓ | ✓ | × | × | Constructor raises `ValueError`. |
| ✓ | ✓ | ✓ | ✓ | Task-backed execution with background recovery and steering. |

An in-memory saver satisfies the constructor check and supports local
multi-turn tests. Restart recovery needs a saver whose data survives the
restart. Task registration and recovery depend on Agent Server's task runtime.
The wrapper installs file-backed replay when resilience is enabled; steering
alone uses in-memory replay, except when attaching to a Responses host whose
stream registry is already configured.

## Request combinations

| Request / configuration | Supported | Result |
| --- | :---: | --- |
| Foreground JSON, either execution path | ✓ | HTTP 200 with response text and optional HITL `output`. Task execution does not change the successful JSON shape. |
| Foreground SSE, either execution path | ✓ | Tokens and terminal SSE events; the model/runnable must emit compatible AI message chunks. |
| Background JSON, resilience enabled | ✓ | HTTP 202, then poll by invocation ID. |
| Background JSON, resilience disabled | × | HTTP 400, even when steering is enabled. |
| Background SSE | × | HTTP 400. |
| `previous_invocation_id`, both options off | × | HTTP 400. |
| `previous_invocation_id` matches the latest task input | ✓ | Continues the same session if normal admission succeeds. It does not restore an arbitrary earlier checkpoint. |
| Concurrent turn, task path, steering off | × | HTTP 409 `conversation_locked`. |
| Concurrent turn, task path, steering on | ✓ | Queues the new turn and signals the active turn. Queue capacity errors return HTTP 429. |
| GET/cancel for an unknown invocation, handlers enabled | × | HTTP 404. |

Task statuses include `queued`, `in_progress`, `completed`, `failed`, and
`cancelled`. Cancellation can return `cancelling` while work winds down.
Steering ends the superseded invocation as `cancelled` with error code
`steered`; its waiting foreground JSON request returns HTTP 409. In Responses,
the normal superseded turn instead completes with its checkpointed partial
output.

The default direct path has no wrapper-owned task queue or latest-parent
precondition. Sharing a checkpointed session there does not add those
concurrency guarantees.

Malformed request bodies return HTTP 400. Direct foreground graph failures
return HTTP 500 with `Internal server error.` Direct SSE failures emit an
`error` event containing the exception text; task-backed failures also store
the exception text in their error envelope. Error details therefore differ
from the Responses host's sanitized handler errors.

## State and recovery

| State | Backend / lifetime | Notes |
| --- | --- | --- |
| Session identity | Public `agent_session_id`; internal hash of user ID plus session ID | Task IDs and graph thread IDs use the internal identity. Reuse the same platform user context for later requests. |
| Invocation envelope and recovery data | `FileInvocationStateStore` locally; Foundry State Store when hosted | Created on the task path. Active envelopes have no expiry; terminal envelopes remain retrievable for 30 days. Recovery data includes graph checkpoint references and, when ready, final response/output. |
| Replay events | File-backed with resilience; normally in-memory with steering alone | Replay TTL is 600 seconds. Expired replay does not remove the separately retained terminal result. |
| Graph state | Application-provided LangGraph saver | Required for background execution and durable checkpoint recovery. Independent of invocation-envelope retention. |

On task recovery, the host first returns an already persisted terminal
result. If final response text/output was saved before terminal publication,
it publishes that result without rerunning the graph. Otherwise, a saved
`(thread_id, checkpoint_id)` resumes the graph with `input=None`. With no
saved reference, the host processes the original text or HITL input again.
Task input retains the user/call context needed by the recovered run.

For checkpointed task execution, graph writes use synchronous durability and
checkpoint events update the invocation's recovery reference. The reference
write and external tool effects are separate operations; recovery can repeat
work across an uncommitted boundary. Use replay-safe tools. Foreground SSE
does not gain a public reconnect contract when tasks are enabled.

The supported background recovery setup is illustrated in the
[resilient invocation sample](../../../../samples/hosting/langgraph-hosted-agents/invocations/04_resilient/README.md).
Its persistent local saver is separate from the SDK's invocation status store.

## HITL combinations

String messages and structured resume messages are different request shapes.
A structured `message` must be a non-empty list of `function_call_output` or
`mcp_approval_response` objects. Arbitrary Responses message/image items are
not accepted there.

| Input / execution combination | Supported | Behavior |
| --- | :---: | --- |
| Matching resume item + pending interrupt | ✓ | Works for foreground JSON, foreground SSE, and background polling. |
| Plain string + pending interrupt | ✓ | Re-emits pending items with empty response text, without injecting a new user turn. |
| Structured resume items + no pending interrupt | × | Direct/task-backed foreground JSON returns HTTP 400. Background/task streaming reports failure through its result or event channel. |
| Custom interrupt + negative MCP approval | × | Fails with `interrupt_rejected`; foreground JSON returns HTTP 409. |
| Recognized middleware approval/rejection | ✓ | Uses the same per-action decision validation as Responses. |
| Partial parallel resume | ✓ | Only outstanding interrupts are emitted again. |
| Conflicting approvals or disallowed decisions | × | Fails the request before resuming any interrupt. |

The rich resume format, approval IDs, and precedence rules are shared with
[Responses HITL](responses_protocol.md#hitl-combinations). A valid rich
`function_call_output` wins over an approval for the same interrupt. Custom
`build_input` hooks can also return a `Command` when a graph is paused.

## Implementation and verification references

- [Host, task path, and SSE conversion](../../langchain_azure_ai/agents/hosting/_invoke_host.py),
  [invocation persistence](../../langchain_azure_ai/agents/hosting/_invocation_store.py),
  and [CLI](../../langchain_azure_ai/agents/hosting/run.py).
- [Invocation behavior tests](../../tests/unit_tests/agents/hosting/test_invoke_host.py),
  [retention tests](../../tests/unit_tests/agents/hosting/test_invocation_store.py),
  and [shared HITL tests](../../tests/unit_tests/agents/hosting/hitl/).
- [Invocation samples](../../../../samples/hosting/langgraph-hosted-agents/README.md#invocations-protocol).

These are source and test references, not a claim that every combination was
executed against a live Foundry deployment during this documentation review.
