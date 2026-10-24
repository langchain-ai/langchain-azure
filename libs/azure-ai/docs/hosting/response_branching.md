# Response Branching

Response branching lets you send new input from the saved graph state of an
earlier response, called the **parent**. Use it to continue from an answer, try
a different direction, or regenerate an answer.

It is available only through `ResponsesHostServer` and is disabled by default.
**Response-ID branching requires a stored parent and its exact graph checkpoint.**
The server must have an explicitly configured response provider.

> **Specification status:** This document defines the target contract. Requiring
> an explicit provider and the OpenAI compatibility requirements below are
> specification changes, not claims that the current implementation supports
> them. See [Implementation gaps and verification](#implementation-gaps-and-verification).

## OpenAI compatibility target

Public HTTP request and response behavior under `/responses` must follow the
[OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)
and [conversation-state guide](https://developers.openai.com/api/docs/guides/conversation-state).
The [OpenAI quickstart](https://developers.openai.com/api/docs/quickstart?tool-type=web-search)
provides the client usage baseline; the API reference defines the field semantics.
This does not add OpenAI-hosted models or built-in tools to the configured graph.

This specification does not require the default host to expose `/conversations`
or its item-management routes. The `/responses` request field `conversation`
references an existing conversation resolved by the configured provider or
platform. Creating and managing that resource is outside this API path's scope.

> **Current limitation:** Branching from a parent with outstanding function
> calls (`function_call`) or MCP approval requests (`mcp_approval_request`) is
> not supported yet. Providing `function_call_output` or `mcp_approval_response`
> does not enable branching from that parent. Support is planned for a future
> update. The tool/approval branching behavior described below is a compatibility
> target, not a current capability. Ordinary tool and approval continuation
> behavior is unchanged.

- `previous_response_id` selects a specific response, not a thread's latest state.
- `conversation` identifies a separate persistent conversation resource. It is
  not a response ID, although a sequential conversation may internally use its
  latest response's checkpoint.
- An existing `conversation` ID continues that conversation's current retained
  state. An unknown or unresolvable ID must produce an explicit error, not create
  a conversation or silently start a new thread. This is the compatibility
  target; current local providers differ as described under
  [Conversation lookup](#conversation-lookup-current-behavior-and-target).
- `previous_response_id` and `conversation` are mutually exclusive.
- Request `store` controls response retention and defaults to `True`. Enabling
  branching must not silently override `store=False`. See
  [Request store behavior](#request-store-behavior) for conversation advancement,
  tool and approval replies, and checkpoint cleanup.
- Tool outputs and approval decisions are valid continuation inputs. A graph
  pause alone must not cause an otherwise valid protocol request to be rejected.
- With `previous_response_id`, a previous response's top-level `instructions`
  are not carried over. Clients resend instructions they want on each turn.

Checkpoint selection is an internal execution strategy. It must preserve these
public semantics rather than introduce a separate client protocol.

## The basic idea

A **checkpoint** is saved LangGraph execution state, including messages and
other graph state. Each reusable response points to its own final checkpoint.

Suppose A suggests a destination. You can ask for two different plans from A:

```text
A: Suggest a destination.
|-- B: Plan a busy itinerary.
`-- C: Plan a relaxed itinerary.
```

Both B and C use `previous_response_id=A.id`. Each starts from A's exact
checkpoint, even if B has already finished when C starts. C does not inherit
B's changes, and A's saved state stays unchanged.

The branches share a LangGraph thread, with separate checkpoint histories.
For response-ID branching, the host restores graph state directly; it does not
rebuild it from response message history.

For internal state selection, a conversation can be treated as a **moving
response reference**, while a response ID is a **fixed reference**:

| Reference                      | Starting state                                                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `previous_response_id=A.id`    | A's exact saved state, regardless of later responses.                                                                    |
| `conversation=conversation.id` | The conversation's current retained state; in a sequential exchange, normally its latest retained response's checkpoint. |

If a conversation advances through A, C, and D, its reference moves forward to D.
Referencing A directly still starts from A. Both linkage forms can use the same
internal state-selection model.

A stored response created within a conversation is also a fixed reference when
selected by `previous_response_id`. Its conversation association does not force
the child to start from the conversation's latest state. A response-ID branch
does not advance that conversation's moving reference.

```text
Conversation: A ---- C ---- D
              |
              `---- B (previous_response_id=A.id)
```

C uses `conversation=conversation.id` after B finishes. It continues from A,
without B's input or output, and advances the conversation to C when stored.
D then continues from C. Creating B neither blocks nor advances the conversation.

A conversation turn with `store=False` reads the selected state but does not
move the reference. Later conversation turns continue from the previously
retained state, without that unretained turn's input, output, or graph changes.

This abstraction preserves the public API: a conversation remains a separate
resource, and the linkage fields remain mutually exclusive. The latest retained
response's checkpoint is usable only when it represents the conversation context
resolved by the provider or platform.

## Quick start

### 1. Enable branching on the server

Start with the [basic hosting setup](../../README.md#hosting-langgraph-agents-on-microsoft-foundry)
and an already configured `model`:

```python
from azure.ai.agentserver.responses import FileResponseStore
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_azure_ai.agents.hosting import ResponsesHostServer

graph = create_agent(model, tools=[], checkpointer=InMemorySaver())

if __name__ == "__main__":
    ResponsesHostServer(
        graph,
        enable_response_branching=True,
        store=FileResponseStore(),
    ).run(port=8088)
```

With branching enabled, `store` must be an explicitly supplied
`ResponseProviderProtocol` implementation. Omitting it or passing `None` raises
`ValueError` during initialization; the host must not silently select a default
provider. This parameter is not a boolean: `store=True` and `store=False` are
not supported constructor values.

`InMemorySaver` is for this local demo only. Its checkpoints disappear on
restart, even if response records survive. Use a durable, history-preserving
saver when state must survive process recreation.

### 2. Create a parent and two branches

Run this in a separate client process:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8088", api_key="local", max_retries=0)

a = client.responses.create(input="Suggest a destination.", store=True)
b = client.responses.create(
    input="Plan a busy itinerary.", previous_response_id=a.id
)
c = client.responses.create(
    input="Plan a relaxed itinerary.", previous_response_id=a.id
)

print(b.output_text)
print(c.output_text)
```

Requests default to `store=True`; A makes this explicit because it will be a
parent. Do not set `store=False` on responses you want to reuse by response ID.
The current response may use `store=False` even when it references a stored
parent; that does not disable the server's provider or delete the parent.
Here, B and C are independent branches from A. C can also start while B is
still running, provided A has finished.

### 3. Continue, regenerate, or rewind

- **Continue B:** send new input with `previous_response_id=b.id`.
- **Regenerate B:** resend B's original input with B's original parent:

```python
regenerated_b = client.responses.create(
    input="Plan a busy itinerary.",
    previous_response_id=b.previous_response_id,
)
```

Every call creates a new response. Regeneration can produce different output.
For client-side **undo/rewind**, select an earlier parent. This does not delete
history, replay stored SSE events, or undo external effects.

### 4. Start or continue a conversation

**Current local host behavior:** The host temporarily accepts a previously
unused, valid `conversation` ID to start a new conversation with empty history.
With `InMemoryResponseProvider` or `FileResponseStore`, you can generate an ID
and reuse it on successive `/responses` requests:

```python
from uuid import uuid4

conversation_id = f"conv_{uuid4().hex}"

first = client.responses.create(
    conversation=conversation_id,
    input="Suggest a destination.",
)
next_turn = client.responses.create(
    conversation={"id": conversation_id},
    input="Plan a relaxed itinerary.",
)
```

**Difference from OpenAI:** OpenAI's `POST /responses` does not create a
conversation from an unknown ID. First create the conversation through
`POST /conversations` (`client.conversations.create()`), then pass its returned
ID as `conversation` to `/responses` for subsequent turns. This separate creation
step is required by OpenAI; the local first-use behavior above is temporary and
does not imply that this host exposes `/conversations` management routes.

`FoundryStorageProvider` and custom providers may reject unknown IDs; use an
existing conversation in their authorized storage scope. See
[Conversation lookup](#conversation-lookup-current-behavior-and-target) for the
current provider differences and the target requirement to reject missing IDs.

The string and object forms of `conversation` refer to the same resource.
Conversation items precede the new input; the response's input and output items
are added to that conversation after it completes with `store=True` (the
default). In a sequential exchange, the next turn therefore includes the
preceding retained turn. With `store=False`, the request can read that history,
but its own turn is not added and the conversation's state does not advance.

You can also branch from a stored response in that conversation, even after the
conversation has advanced:

```python
branch = client.responses.create(
    previous_response_id=first.id,
    input="Plan a busy itinerary.",
)
```

This starts from `first` rather than `next_turn` and does not move the
conversation's head. A later request using `conversation_id` continues from
`next_turn`, not from `branch`.

## Requirements and request rules

> **Current scope:** Branching requires a finished parent graph with no pending
> tasks, interrupts, function calls, or MCP approval requests. Parents with
> outstanding `function_call` or `mcp_approval_request` items remain ineligible
> even when matching replies are supplied. Support for these branching flows is
> deferred to the future scope below. Ordinary tool and approval continuation
> behavior is unchanged.

A response-ID parent is reusable only when:

1. It was stored with `status="completed"` and is available to the requesting caller.
2. Its stored internal metadata identifies its exact final graph checkpoint.
3. That checkpoint and the history needed to restore it are still available.
4. Its graph has finished, with no pending tasks or interrupts, and its output
   has no outstanding function calls or MCP approval requests.

Parents created before branching was enabled or through ordinary continuation,
including responses saved in a conversation, are eligible when they satisfy
these checks. No branching-specific marker is required. Advancing the
conversation does not change the checkpoint selected by an earlier response ID.

**Response completion and graph completion are different.** A response can
contain tool calls or approval requests while its graph is paused. Such a parent
is outside the current branching scope, even if its response status is
`completed`. Current branching restores a finished checkpoint.

### Conversation lookup: current behavior and target

OpenAI's `conversation` field references an existing conversation resource.
Supplying an ID to `POST /responses` does not create that resource. The current
local host does not yet enforce this distinction:

| Case                                                              | Current local behavior                                                                                                                                                                                                              | OpenAI-compatible target                                                                                                                  |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| A previously unused, syntactically valid conversation ID          | `InMemoryResponseProvider` and `FileResponseStore` return empty history. The adapter does not independently check resource existence; without a saved checkpoint, it uses the ID as the graph thread and can start fresh execution. | Reject the missing reference before graph execution, including when `store=False`. Do not create a conversation or substitute a new root. |
| An existing conversation with no retained items                   | Empty history follows the same local path as a missing ID.                                                                                                                                                                          | Accept the existing empty resource and start its first turn. Empty history alone cannot establish whether the resource exists.            |
| `enable_response_branching=True` with an explicit conversation ID | The request still uses ordinary conversation continuation, so enabling branching does not add missing-conversation validation.                                                                                                      | Apply the same conversation existence and authorization checks regardless of the branching flag.                                          |

With `FoundryStorageProvider`, history lookup is delegated to the remote service.
In the inspected Agent Server SDK 2.3.0, reference lookup happens before the graph
handler and backend not-found errors are surfaced to the caller. Local
first-use acceptance must not be assumed for Foundry or custom providers; their
lookup and error behavior must be checked separately.

Direct Azure Responses REST requests with an unknown, valid-format conversation
ID returned HTTP 404 with both `store=True` and `store=False`. The OpenAI
documentation establishes the existing-resource model; exact error details at
OpenAI's public endpoint remain a separate compatibility verification item.

The adapter must obtain an existence-aware result from the authorized provider
or platform. A missing resource, an existing empty resource, an authorization
failure, and a storage failure must not all become an empty history list. The
current local first-use behavior is an implementation difference, not a promised
extension to the target contract.

### Instructions, checkpoints, and routing

Do not inherit a previous response's top-level `instructions` through a graph
checkpoint. Distinguish request-scoped instructions from actual system or
developer message items supplied in conversation history.

The graph's saver must preserve checkpoint history and implement asynchronous
reads and writes. Keep referenced checkpoints and any history needed to restore
them, not just the latest checkpoint.

With `enable_response_branching=True`, request routing is:

| Request linkage                                      | Behavior                                                                                                                            |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Neither parent nor `conversation`                    | Start a new root response in a new thread.                                                                                          |
| `previous_response_id` only                          | Start from the parent's exact checkpoint.                                                                                           |
| Existing `conversation`, without a previous response | Start from its existing items and the new input. An empty conversation starts a fresh root.                                         |
| Existing `conversation`, with previous responses     | Continue its retained context. For sequential turns, use the latest retained response's checkpoint when it represents that context. |
| Unknown or unresolvable `conversation`               | Reject before graph execution; do not create a conversation or substitute a root response.                                          |
| Both parent and `conversation`                       | Reject with HTTP 400.                                                                                                               |

Omitted or `null` linkage fields do not select a parent or conversation.
Resolve references within the request's authorized storage scope; a resource ID
from another scope is not sufficient. Authorization errors and storage failures
must not trigger conversation creation.
The checkpoint shortcut must represent the conversation context resolved by the
provider or platform. Internal parent resolution must not rewrite the
client-visible `previous_response_id` into a response ID for a conversation request.

Keep `options.steerable_conversations=False` and `app=None`. Steering hands off
an in-flight turn; checkpoint-based branching selects a saved response boundary.
The adapter must create its own host to configure branching. Unsupported
combinations, or a saver without the required async methods, raise `ValueError`
during initialization.

The branching flag controls the checkpoint execution strategy for supported
requests. It must not change the public meanings of `conversation`, `store`,
or `instructions`.
With branching disabled, the constructor's existing optional-provider default
remains unchanged.

### Future scope: function calls and MCP approvals

These rules describe planned support for branching from parents awaiting tool
results or MCP approval decisions. They are not part of the current branching
scope. Future support must restore the selected pending state and apply replies
without changing the saved parent or another branch.

- Match a `function_call_output` to its function call using `call_id`, following
  the [function-calling guide](https://developers.openai.com/api/docs/guides/function-calling).
- Match an `mcp_approval_response` to the approval request's `id` using
  `approval_request_id`; preserve the `approve` decision, including rejection.
  The [MCP guide](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)
  explicitly demonstrates this continuation with `previous_response_id`.
- Once supported, do not reject these input types merely because branching is
  enabled. Validate them against the selected response or conversation context
  before graph work.
- Do not infer that all outstanding MCP approvals must be answered in one
  request from the single-approval examples. Partial, missing, duplicate, and
  unknown-reply behavior requires the compatibility verification below.

For response-ID continuation, the selected parent must have been retained（`store=True`）.
An unretained tool or approval response instead requires full input replay and
developer-provided graph support, as described in the store rules below.

## Request `store` behavior

Request `store` is a boolean and defaults to `True`. It controls retention of the
current response, independently of the server's provider configuration. Azure
OpenAI behavior verified through REST is the OpenAI-compatible baseline for the
following public rules.

### Response retention and conversation advancement

For a successfully completed response:

| Linkage                        | Request `store`   | Current response retained? | Effect on the conversation                                                               |
| ------------------------------ | ----------------- | -------------------------- | ---------------------------------------------------------------------------------------- |
| `conversation=conversation.id` | `True` or omitted | Yes                        | Append this turn's input and output, and advance the conversation's head.                |
| `conversation=conversation.id` | `False`           | No                         | Read retained context without appending this turn or advancing the head.                 |
| `previous_response_id=A.id`    | `True` or omitted | Yes                        | Leave A's associated conversation unchanged; the child is a separate response-ID branch. |
| `previous_response_id=A.id`    | `False`           | No                         | Leave the stored parent and its associated conversation unchanged.                       |

The same retention rules apply to a root response without either linkage field.
Context selection is independent of retention: a conversation request reads its
current retained context, while a response-ID request reads its selected parent's
exact state. `store=False` does not delete an existing parent or conversation
history. The next conversation request still reads the preceding retained state,
excluding the unretained turn.

An unretained response is unavailable for retrieval (HTTP 404). Referencing it
through `previous_response_id` fails with HTTP 400,
`code="previous_response_not_found"`, and `param="previous_response_id"`.
Returning a function call or MCP approval request does not make an exception.

### Function-call outputs and MCP approval replies

An unretained response A can return `function_call` or `mcp_approval_request`
items, including with response `status="completed"`. The client can still submit
a matching reply, but cannot rely on server-side retention of A:

| Follow-up to A created with `store=False`                                              | Function-call result                                      | MCP approval reply                                                     |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------- |
| `previous_response_id=A.id` plus the reply                                             | HTTP 400, `previous_response_not_found`.                  | HTTP 400, `previous_response_not_found`.                               |
| Only the reply, without A's call/request in input                                      | HTTP 400; no matching function call.                      | HTTP 400; the approval request is missing from input.                  |
| Original input, A's complete output, and the matching reply; no `previous_response_id` | Accepted; the model can use the supplied function result. | Accepted; an approved MCP call can execute and produce a final answer. |

Full input replay has this form:

```text
POST /responses
store=false
input = original request input + all items from A.output + matching reply
```

The matching reply is a `function_call_output` with the original `call_id`, or an
`mcp_approval_response` with the original `approval_request_id` and `approve`
decision. Resend the required tool definitions and current request instructions.
Include the complete prior output, not only the reply identifier. A `call_id`
or approval-request ID alone does not restore the missing context.

Conversely, when A was created with `store=True`, a follow-up can use
`previous_response_id=A.id` and submit its matching reply even if the follow-up
itself uses `store=False`. Parent retention and child retention are independent.

**Developer responsibility:** To use full input replay, the application developer
must adapt the graph's input handling, routing, and state reconstruction to
consume replayed calls, results, approval requests, and decisions as a new
execution, without relying on a saved interrupt or checkpoint from A. The client
supplies the complete context; the graph must know how to process it.
`ResponsesHostServer` does not automatically make an arbitrary graph replay-capable and must not resume A's deleted checkpoint.

The accepted replay cases above describe the protocol contract, not a guarantee
for an unmodified graph. This responsibility does not expand the current
finished-parent branching scope.

### Checkpointer isolation and cleanup

The retention and advancement rules apply with or without a LangGraph
checkpointer, regardless of `enable_response_branching`. With `store=False`,
preserve the conversation's retained checkpoint reference and all prior graph
state. This isolation covers all graph state, not only messages or SDK history.

For this host, an unretained response's execution-owned checkpoints are temporary:

- Checkpoints may be written while the request executes or is awaiting supported recovery. They must never become the retained conversation head.
- Once execution is finalized, delete host-managed checkpoints and pending writes owned exclusively by that unretained response. This also applies when the response ends by returning a function call or MCP approval request, or reaches terminal failure or cancellation.
- Delete only this response's data. Preserve stored parents, sibling branches, and checkpoints needed to restore retained state; do not delete a shared thread.
- An execution deferred for supported recovery is not finalized. Cleanup must not remove the progress needed for that recovery; recovered finalization applies the same `store=False` cleanup rule.
- Surface cleanup failures explicitly; do not silently report successful cleanup.

A later conversation request must resolve its retained head, not the saver's
implicit latest checkpoint. If the existing conversation was empty before the
unretained turn, it must still start empty afterward. Checkpoint deletion is a
host lifecycle requirement, not a claim about OpenAI's internal storage design.

## What must be stored?

Branching needs **both** response records and graph checkpoints:

| Data                                                               | Where it lives             | How to configure it                                |
| ------------------------------------------------------------------ | -------------------------- | -------------------------------------------------- |
| Response, status, input/output, and internal checkpoint references | SDK response provider      | Server: `ResponsesHostServer(..., store=provider)` |
| Graph state, pending writes, and checkpoint history                | LangGraph checkpoint saver | Graph: `checkpointer=...`                          |

With branching enabled, the server's `store` is a required explicit provider;
it configures storage capability and the backend. The client's `store` is a
boolean that defaults to `True` and controls retention of that response. These
are separate contracts: having a provider does not force every response to be
retained, and a client setting `store=True` cannot supply a missing provider.

Unretained responses follow the
[request store rules](#request-store-behavior), including execution-owned
checkpoint cleanup.

Use an appropriate provider explicitly, such as `FileResponseStore` for a local
demo or `FoundryStorageProvider` on Foundry. Response retention does not replace
the graph's checkpointer. The application owns the graph saver's initialization,
lifecycle, and general retention policy; the host must implement the unretained
execution cleanup specified above.

**Migration:** Branching callers that relied on automatic provider selection
must pass that provider explicitly. Preserve its backend configuration and
existing data; selecting a different provider does not migrate response records
or graph checkpoints.

**Retaining a response does not retain or renew its graph checkpoints.**
Durable backends can still expire data. Align both lifetimes, retain history
needed by recoverable tasks, and clean up unneeded data through the backends.
For retained responses, the host does not coordinate retention or cascade response
deletion to checkpoints.
Backend retention policies are separate from OpenAI's hosted data-retention
guarantees; this spec does not promise an identical retention period.

## Failures and limitations

Check the response status or terminal SSE event, not only the HTTP status:
execution errors can arrive as `status="failed"` or `response.failed` after
HTTP admission. Parameter and reference validation must prevent graph execution;
do not turn those caller errors into successful-looking graph runs. Match the
OpenAI HTTP error and streaming contracts for the corresponding condition.
Backend execution failures use public error code `server_error`; internal codes
are logged. Exact reference-error statuses and codes need the verification below.

| Problem                                                                                                             | Result and action                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Request includes `response_id`, an invalid parent ID, or both linkage fields                                        | HTTP 400 before execution. Omit `response_id`; use a nonempty string or `null` for the parent, and choose one linkage type. |
| Referenced response is missing or inaccessible                                                                      | Return the corresponding OpenAI-compatible reference error before graph execution. Do not substitute a new root response.   |
| `previous_response_id` refers to an unretained response                                                             | HTTP 400 with `previous_response_not_found`; do not restore an internal checkpoint in place of the missing response.        |
| Conversation ID is unknown or malformed, cannot be resolved, or lookup fails due to authorization or storage errors | Return the appropriate error before graph execution. Do not create a replacement conversation or start a new thread.        |
| Stored parent lacks a checkpoint reference or its graph state cannot be restored                                    | Backend failure. Restore the required data or choose another parent.                                                        |
| Input contains approval or tool-result replies                                                                      | Validate the linked calls and use the appropriate continuation path; these input types are not blanket errors.              |
| Exact checkpoint is missing or mismatched                                                                           | Failed response. Restore the required data or choose another parent; there is no fallback to the latest checkpoint.         |
| Graph checkpoint write fails                                                                                        | Failed response. Progress is not published for the missing checkpoint; investigate the saver backend.                       |
| SDK terminal response storage fails                                                                                 | The graph checkpoint alone cannot serve as a parent. Check SDK storage errors and the stored response status.               |

- Only single-worker execution is supported. A durable saver supports process
  recreation, not distributed execution or guarantees for actual process-crash
  windows.
- The host selects only the root checkpoint namespace (`checkpoint_ns=""`).
  Private persistent subgraph histories are not rewound or isolated by the host.
- External effects are not rolled back or guaranteed to happen exactly once.
  Applications must make replayed work safe.

## Background recovery

The Agent Server SDK controls when recovery runs. Enabling branching does not
itself enable background recovery. Resolve and persist the effective parent or
conversation snapshot before graph execution. A recovered conversation request
must not select a newer conversation head.

The current checkpoint recovery machinery has the following behavior; these
backend limits are not additional OpenAI request-validation rules:

| Saved state                                                               | Recovery behavior                                                                                                                         |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Root response, with or without saved progress                             | Fail; submit a new root request.                                                                                                          |
| Child with no saved progress                                              | Load and validate the parent's exact checkpoint, then replay the child's input. Fail if the parent or required checkpoint is unavailable. |
| Child with valid, response-owned progress                                 | Resume that checkpoint with input `None`, preserving pending writes. No parent response lookup is needed.                                 |
| Child's saved graph is already finished                                   | Finish response finalization without running graph nodes again.                                                                           |
| Malformed progress, unavailable/mismatched checkpoint, or wrong ownership | Fail; do not fall back to the parent or the thread's latest checkpoint.                                                                   |

Recovery uses the last response snapshot the SDK actually persisted. If graph
execution advanced beyond that snapshot, work and external effects can repeat.
Even with a finished graph checkpoint, response finalization and storage may run
again. Keep the checkpoint history needed to restore the child's state.

## Implementation notes

The following details are for maintainers; applications do not need to manage
these fields or call these helpers.

The execution flow below is limited to finished-parent branching. Branching from
parents awaiting function results or MCP approval decisions is deferred to
[Future scope](#future-scope-function-calls-and-mcp-approvals).

### Checkpoint references

| Name             | Meaning                                                                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Origin           | The parent's exact checkpoint selected before a child starts. No separate origin record is stored.                                                |
| Progress         | This response's latest confirmed graph checkpoint, referenced by SDK recovery metadata.                                                           |
| Final checkpoint | The last progress reference stored with a completed response. Parent eligibility is checked against the restored graph state and response output. |

These are roles for references, not copies of graph state or three separate
records. A root response has no parent; both root and child responses use the
root checkpoint namespace.

### Fresh request

1. **Select.** Resolve the starting state using the platform's authorization
   context. An explicit parent uses `select_branch_checkpoint`; a conversation
   ID resolves to its existing response/checkpoint reference. An unknown or
   unresolvable conversation fails before execution.
   `branch_selection_scope` caches response-ID selections only for this request.
   Persist the resolved starting reference for recovery before execution.
2. **Validate.** For a selected parent, verify that its stored response is
   completed and has no outstanding function calls or MCP approval requests.
   After configuration hooks and graph-bound configuration merging, re-pin the
   parent's thread, root namespace, and checkpoint ID. Read through
   `validate_branch_checkpoint` and `ResponseCheckpointSaver`. The returned tuple
   and payload must match the selected identity, with no next tasks or interrupts.
   Validation failure prevents execution.
3. **Execute and save.** Shallow-copy the graph and attach a request-local saver
   wrapper; leave the application's graph and saver unchanged. Pass current input
   and instructions, using the checkpoint for prior state. Run with
   `durability="sync"` and `updates`, `messages`, and `checkpoints` stream modes.
4. **Publish progress.** For each graph checkpoint event, await
   `wait_for_checkpoint` until the underlying saver write succeeds. Then update
   `TaskStorageManager`, flush output events, and emit `stream.checkpoint()`.
   A graph event is not proof of persistence; an SDK task checkpoint requests SDK
   persistence and is not a successful-write acknowledgement.
5. **Finish.** After shutdown/cancellation checks and pending interrupt emission,
   emit `response.completed` for SDK finalization. Request `store` controls
   terminal response retention. A retained response keeps the same confirmed
   final checkpoint reference; no separate boundary is written. An unretained
   response must complete execution-owned checkpoint cleanup as specified under
   [Checkpointer isolation and cleanup](#checkpointer-isolation-and-cleanup).

Graph writes and SDK response storage are **not a transaction**. A graph
checkpoint alone cannot make a reusable parent: its response must also be stored
and completed, and its output and restored graph state must satisfy the current
finished-parent requirements. A response whose graph pauses for a tool result or
approval is not a reusable parent within this scope.

### Conversation head references

Conversation continuation needs a mapping from `conversation_id` to its current
response/checkpoint reference. The existing `conversation_chain_store` may be
reused for that mapping; a full conversation chain is not required for branching.

Response-ID branching selects the exact reference from the stored parent response
without consulting the mutable conversation head. A conversation request resolves
its head once and persists that starting reference for recovery. Later head
updates must not redirect an in-flight or recovered request.

For a completed conversation turn, advance the head to its confirmed final
reference only when `store=True` and both its response record and checkpoint
are persisted. With `store=False`, leave the head unchanged even if execution
or recovery wrote newer graph checkpoints. A subsequent conversation request
must resolve the retained head, not the saver's implicit latest checkpoint.
Response-ID branches do not update the conversation head.

Every conversation-head write path must enforce request `store`, including
paths for pauses, shutdown, cancellation, and recovery finalization. Internal
progress references remain separate from the retained conversation head.
If an existing conversation has no retained checkpoint yet, a `store=False`
turn must leave that starting state empty. The next request must not discover
and restore the unretained turn through a thread-level latest-checkpoint lookup.

### Internal metadata and isolation

- SDK progress fields: `langgraph_thread_id` and `langgraph_checkpoint_id`.
- Admitted mode: `langgraph_response_branching="checkpoint-v1"`. Recovery uses
  the original mode, not the current server flag. Middleware strips client mode
  headers; inconsistent recovery mode fails.
- Graph ownership: `langgraph_response_execution` stores the JSON-encoded array
  `[response_key, parent_response_id]`; the parent is `null` for a root response.
  Private subgraph checkpoints are not stamped.
- A conversation's resolved parent response/checkpoint reference must be retained
  separately from the public request fields. Recovery must use that admitted
  reference, not whatever response is latest when recovery starts.
- A root thread uses the response key: the response ID prefixed with
  `user-<SHA-256 hex digest of user_id_key>:` when a nonempty platform
  `user_id_key` is present. This is separate from backend `user_isolation`.

### Recovery details

After configuration hooks and saver setup, validate that saved child progress
belongs to both the current response key and the parent resolved for that request.
Matching the thread alone is insufficient. Valid progress needs no parent lookup;
absent progress requires the same admitted starting reference, not a fresh
selection of a mutable conversation head.

`ResponseCheckpointSaver.resume_config` removes the outward `checkpoint_id`
for native resume, retaining the exact reference as a request-local recovery
head. Root reads and writes without an explicit ID use that head. It advances
only after a successful save with the expected thread, namespace, and new ID.
Pending writes retain task IDs and supported task paths, so a newer sibling
cannot redirect recovery reads or receive this response's writes.

The stream starts from persisted output. The converter restores missing text
from completed nodes' cached updates and records emitted message IDs to avoid
appending persisted text again.

### Source entry points

- [Responses handler](../../langchain_azure_ai/agents/hosting/_responses_host.py):
  `build_runnable_config`, `handle_create`, and `_new_stream`.
- [Branch selection and saver wrapper](../../langchain_azure_ai/agents/hosting/_responses/branching.py):
  selection, exact reads, ownership, persistence barriers, and recovery heads.
- [Graph-stream conversion](../../langchain_azure_ai/agents/hosting/_converters/_stream.py):
  graph persistence before SDK progress publication.
- [Branching tests](../../tests/unit_tests/agents/hosting/test_response_branching.py):
  existing parent preservation, concurrent sibling, failure, and recovery coverage.
  The target contract below requires additional compatibility coverage.

The branching helpers and saver wrapper are internal, not public extension points.

## Implementation gaps and verification

This revision changes the specification only. In particular:

- The current constructor selects a default provider when branching is enabled
  and `store` is omitted. It must be changed to require an explicit provider.
- The current branching path supports finished parents and rejects approval and
  tool-result replies. Supporting pending-parent continuation requires host
  implementation work. Full input replay for unretained tool and approval
  responses is a separate application responsibility: developers must adapt
  their graphs as described under
  [Function-call outputs and MCP approval replies](#function-call-outputs-and-mcp-approval-replies).
- Explicit conversations currently use the ordinary continuation path. Verify
  authorized context resolution, checkpoint selection, and public response fields
  against the conversation contract above, including response-ID selection of a
  stored conversation response and isolation of unretained turns. Implement
  existence-aware lookup as described under
  [Conversation lookup](#conversation-lookup-current-behavior-and-target), and
  prevent `store=False` turns from updating the retained conversation head on
  every checkpointed continuation and finalization path.
- Ordinary continuation currently updates its latest checkpoint reference
  without checking request `store`, including explicit conversation requests
  when branching is enabled. Fix this compatibility gap and implement cleanup
  of finalized unretained executions; neither is an optional checkpointer
  exception to the store contract.
- Checkpoint restoration must be verified not to inherit request-scoped
  `instructions` from the parent.

Conversation advancement and response-ID branch isolation were verified through
REST at an Azure project-scoped OpenAI endpoint. Unretained function-call and MCP
approval continuations were verified at the account-level OpenAI endpoint.
These results define the compatibility baseline above; they do not establish
support in the host implementation.

Before claiming compatibility, cover these observable scenarios:

- A valid explicit provider and a history-preserving saver enable branching;
  missing providers and boolean constructor values are rejected.
- B and C can select A independently, including when a newer sibling exists.
  A child with `store=False` does not delete or modify its stored parent.
- A stored, eligible response created in a conversation can be selected using
  only `previous_response_id`, even after that conversation advances. Its branch
  uses the selected response's context and leaves the conversation head unchanged.
  Subsequent conversation turns can continue and advance without inheriting
  the branch's input or output.
- A conversation turn with `store=False` can read retained context but is not
  retrievable or reusable by response ID. Its input and output are not appended
  to the conversation, and the next conversation turn still starts from the
  retained head, even when the unretained execution wrote newer checkpoints.
  Verify this through `POST /responses` with a real graph saver, both with
  branching enabled and disabled: after retained A, unretained B must leave A's
  head unchanged, and a subsequent stored C must read A without B's messages
  or other graph-state changes. Also verify that an unretained first turn in an
  existing empty conversation leaves the next turn starting empty. Repeat the
  isolation checks for supported recovery and interrupted-finalization paths.
- For an unretained response ending in a function call or MCP approval request,
  reject response-ID continuation and a reply without matching input history.
  Verify full input replay with the matching reply using a developer-adapted,
  replay-capable graph, without restoring the deleted checkpoint. This is not
  a guarantee for arbitrary graphs. Also verify the converse: a stored parent
  accepts a reply whose child response uses `store=False`. Check function results
  and actual execution of approved MCP calls, not only request admission.
  Pending-parent branching remains in future scope.
- After an unretained execution is finalized, its host-managed checkpoints and
  pending writes are removed without deleting or changing retained parents or
  siblings. Cover tool/approval outputs, terminal failure/cancellation, supported
  recovery, and explicit cleanup errors.
- Resolve an existing conversation through the configured provider or platform,
  reuse both ID forms, and verify that stored turns advance its retained context.
  Also cover an existing empty conversation and recovery after the conversation
  advances.
- Unknown, malformed, or unresolvable conversation IDs, authorization errors,
  and storage lookup failures must not create conversations or start graph
  execution. Check valid but absent IDs with both `store=True` and `store=False`
  through `POST /responses`, covering local memory/file providers and the
  Foundry-backed path. Thread-mapping helper tests alone do not establish
  reference-validation behavior.
- Return matched function outputs and approved or denied MCP decisions through
  the same linkage used for ordinary continuation.
- Verify HTTP errors, error objects, terminal SSE events, and absence of graph
  execution on caller-input failures, not only internal helper results.

The cited documentation and service verification do not establish every edge
case. Confirm these remaining cases against the service before claiming
compatibility:

- Exact status, error code, and parameter attribution for reference errors not
  already specified above, including inaccessible resources and malformed
  response or conversation references.
- Acceptance of parents in different response statuses.
- Partial, missing, duplicate, or unknown tool outputs and approval decisions,
  including requests with multiple outstanding calls. Do not assume function
  outputs and MCP approvals have identical batch-completeness rules.
- Concurrent requests to one conversation and the visibility/order of its items.

These are compatibility verification items, not permission to invent stricter
branching-only rules or silently fall back to a different conversation or parent.
