# Response Branching

Response branching lets you send new input using the context of an earlier
response, called the **parent**. Use it to continue from an answer, try a
different direction, or regenerate an answer.

Checkpoint-based branching is available through `ResponsesHostServer` and is
disabled by default.
**It requires a stored parent and its exact graph checkpoint.**
The server must have an explicitly configured **response storage provider** for
response records, separate from the graph checkpointer.

> **Specification status:** This document defines the target contract, not a
> claim of full implementation. Current behavior, implementation gaps, and
> deferred scope are listed under
> [Implementation gaps and verification](#implementation-gaps-and-verification).

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

- `previous_response_id` selects a specific response, not a thread's latest state.
- `conversation` identifies a separate persistent conversation resource. It is
  not a response ID, although a sequential conversation may internally use its
  latest response's checkpoint.
- An existing `conversation` ID continues that conversation's current retained
  state. An unknown or unresolvable ID must produce an explicit error, not create
  a conversation or silently start a new thread. See
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

### Checkpoint-based branching

With a graph checkpointer, the host selects `langgraph_checkpoint` mode: prior
state comes from a checkpoint, so only the current request input is passed to
the graph. The checkpoint-specific eligibility, storage, and recovery rules
below describe this mode, not history-based continuation.

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

Both linkage forms resolve an exact starting checkpoint before entering the
same graph execution path:

| Reference                      | Lookup                                                              | Starting checkpoint                                          |
| ------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------ |
| `previous_response_id=A.id`    | Load A from the response store and read its checkpoint reference.   | A's exact final checkpoint, regardless of later responses.   |
| `conversation=conversation.id` | Read the conversation's head reference from the conversation store. | The final checkpoint of its latest completed, retained turn. |

A response ID is a **fixed checkpoint reference**. A conversation ID is a
**moving checkpoint reference**, advanced only after a retained turn completes.
Intermediate checkpoints are execution/recovery progress, not conversation heads.

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

Starting from the [basic hosting setup](../../README.md#hosting-langgraph-agents-on-microsoft-foundry)
with a configured `model`, supply an explicit response provider and graph
checkpointer:

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

Server `store` is a response provider, not a boolean. `InMemorySaver` is for
local demos; use a durable, history-preserving saver for reuse after restart.

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

A must be stored (`store=True`, the default) and its graph must finish.
B and C each start from A's exact checkpoint, independently of one another.

### 3. Continue, regenerate, or rewind

- **Continue B:** send new input with `previous_response_id=b.id`.
- **Regenerate B:** resend B's input with its original parent,
  `previous_response_id=a.id`, to create a new response.
- **Rewind:** select an earlier parent to start a new response; this does not
  delete history, replay stored SSE events, or undo external effects.

## Requirements and request rules

With `enable_response_branching=True`, three requirements are mandatory:

- The graph must have a checkpointer. Reject a graph without one with
  `ValueError` during host initialization.
- A response-ID branch must restore the selected parent's exact final checkpoint,
  not the thread's or conversation's latest checkpoint.
- If the required checkpoint reference or checkpoint is missing, mismatched, or
  unreadable, fail before graph execution. Do not fall back to message-history
  reconstruction, a latest checkpoint, or a new root response.

The parent eligibility rules below cover finished-parent checkpoint branching.
Parents with outstanding function calls or MCP approval requests remain ineligible
even when matching replies are supplied; their support is deferred to
[Future scope](#future-scope-function-calls-and-mcp-approvals).
Ordinary tool and approval continuation behavior is unchanged.

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
Supplying an ID to `POST /responses` does not create that resource. Reject a
missing reference before graph execution, including when `store=False`, but
accept an existing empty conversation and start its first turn. Apply the same
existence and authorization checks regardless of the branching flag.

The adapter must obtain an existence-aware result from the authorized provider
or platform. A missing resource, an existing empty resource, an authorization
failure, and a storage failure must not all become an empty history list. The
current provider differences are recorded under
[Implementation gaps and verification](#implementation-gaps-and-verification).

### Instructions, checkpoints, and routing

Do not inherit a previous response's top-level `instructions` through a graph
checkpoint. Distinguish request-scoped instructions from actual system or
developer message items supplied in conversation history.

The graph's saver must preserve checkpoint history and implement asynchronous
reads and writes. Keep referenced checkpoints and any history needed to restore
them, not just the latest checkpoint.

With `enable_response_branching=True`, request routing is:

| Request linkage                                      | Behavior                                                                                    |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Neither parent nor `conversation`                    | Start a new root response in a new thread.                                                  |
| `previous_response_id` only                          | Start from the parent's exact checkpoint.                                                   |
| Existing `conversation`, without a previous response | Start from its existing items and the new input. An empty conversation starts a fresh root. |
| Existing `conversation`, with previous responses     | Load the conversation store's retained final checkpoint and run with the new input.         |
| Unknown or unresolvable `conversation`               | Reject before graph execution; do not create a conversation or substitute a root response.  |
| Both parent and `conversation`                       | Reject with HTTP 400.                                                                       |

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

History-based continuation uses SDK-managed response and history storage.
Checkpoint-based branching additionally needs graph checkpoints, and checkpointed
conversation continuation needs a retained head mapping:

| Data                                                               | Where it lives             | How to configure it                                |
| ------------------------------------------------------------------ | -------------------------- | -------------------------------------------------- |
| Response, status, input/output, and internal checkpoint references | SDK response provider      | Server: `ResponsesHostServer(..., store=provider)` |
| Graph state, pending writes, and checkpoint history                | LangGraph checkpoint saver | Graph: `checkpointer=...`                          |
| Conversation ID -> retained final checkpoint reference             | Conversation store         | Host: existing `conversation_chain_store`          |

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
itself enable background recovery: automatic crash recovery requires SDK opt-in
and stored background work. Root and child responses use the same SDK admission
policy. For the target conversation behavior, resolve and persist the effective
conversation snapshot before graph execution; a recovered conversation request
must not select a newer conversation head.

The current checkpoint recovery machinery has the following behavior; these
backend limits are not additional OpenAI request-validation rules:

| Saved state                                                               | Recovery behavior                                                                                                                         |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Root with no saved progress                                               | Replay its original request using the existing root configuration.                                                                        |
| Child with no saved progress                                              | Load and validate the parent's exact checkpoint, then replay the child's input. Fail if the parent or required checkpoint is unavailable. |
| Root or child with valid, response-owned progress                         | Resume that checkpoint with input `None`, preserving pending writes. No parent response lookup is needed.                                 |
| Root or child's saved graph is already finished                           | Finish response finalization without running graph nodes again.                                                                           |
| Malformed progress, unavailable/mismatched checkpoint, or wrong ownership | Fail; do not fall back to the parent or the thread's latest checkpoint.                                                                   |

Recovery uses the last response snapshot the SDK actually persisted. If graph
execution advanced beyond that snapshot, work and external effects can repeat.
Even with a finished graph checkpoint, response finalization and storage may run
again. Keep the checkpoint history needed to restore the response's state.

## Implementation notes

The following details are for maintainers; applications do not need to manage
these fields or call these helpers.

Branching adds exact starting-state selection to the existing hosting path,
not a separate execution or recovery subsystem. Configuration and input hooks,
graph streaming, output conversion, progress persistence, and SDK finalization
remain shared with ordinary checkpointed execution.

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
   context. `previous_response_id` loads the stored response's exact checkpoint
   reference through `select_branch_checkpoint`; a conversation ID reads its
   retained final checkpoint reference from the conversation store. Both then
   use the same checkpoint-based graph execution path. An unknown or unresolvable
   conversation fails before execution.
   `branch_selection_scope` caches response-ID selections only for this request.
   For the target conversation behavior, persist the resolved starting reference
   for recovery before execution. Response-ID parent selection is request-local;
   a child without saved progress re-reads its stored parent.
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
   Only successful finalization of a stored conversation turn updates its
   conversation-store head, as described below.

Graph writes and SDK response storage are **not a transaction**. A graph
checkpoint alone cannot make a reusable parent: its response must also be stored
and completed, and its output and restored graph state must satisfy the current
finished-parent requirements. A response whose graph pauses for a tool result or
approval is not a reusable parent within this scope.

### Conversation head references

The conversation store maps each authorized `conversation_id` to the exact final
graph checkpoint of its latest completed, retained turn. The existing
`conversation_chain_store` may be reused for this mapping; a full conversation
chain is not required for branching.

Response-ID branching selects the exact reference from the stored parent response
without consulting the mutable conversation head. A conversation request resolves
its head once and persists that starting reference for recovery. Later head
updates must not redirect an in-flight or recovered request.

Keep the previous conversation head unchanged while the graph runs. Intermediate
checkpoints may be published as execution/recovery progress, but never as the
retained head. At successful turn finalization, update the conversation store
to that run's confirmed final checkpoint only when `store=True` and both the
response record and checkpoint are persisted. The next conversation request
starts from this final checkpoint, not the saver's implicit latest checkpoint.
Response-ID branches do not update this head.

Streaming, pauses, shutdown, cancellation, and recovery must not bypass this
finalization rule or promote an intermediate checkpoint. A failed, cancelled,
or unretained turn leaves the prior head unchanged, even if it wrote newer
checkpoints. Internal progress references remain separate from the retained head.
If an existing conversation has no retained checkpoint yet, a `store=False`
turn must leave that starting state empty. The next request must not discover
and restore the unretained turn through a thread-level latest-checkpoint lookup.

### Current continuation pointer

The existing pointer key is the user-scoped SDK `conversation_chain_id`, not
the selected LangGraph thread ID or a global branch-tree head. Forks can share
a graph thread without sharing that SDK key.

Current ordinary continuation writes available progress after graph streaming,
before shutdown/cancellation checks and terminal response persistence. This can
include paused or cancelled work. A later terminal-storage failure does not
roll back that pointer; concurrent writes follow the configured store's ordering.
These existing semantics do not yet implement the retained conversation-head
contract above.

Response-ID branching neither reads nor writes this legacy pointer. The stored
parent and the current response's own progress supply its references, without
a branch index or an additional side-store write.

### Shared checkpoint adaptation

`ResponseCheckpointSaver` lives in the existing task-storage module. It is a
request-local asynchronous hosting adapter shared by branching and ordinary
checkpointed execution, not another storage backend or general-purpose saver.
The application's graph and saver retain storage and lifecycle ownership.

The adapter supplies exact reads, response-owned native recovery, and graph-write
confirmation before SDK progress publication. Synchronous private-subgraph calls
pass through without touching root progress; native subgraph replay delegates
history queries. Serialization and channel-version allocation also remain with
the application saver. Root progress confirmation needs no cross-thread futures
or locks.

Parent selection and eligibility stay in branch preparation. Completion retains
the last confirmed progress reference, without a separate boundary marker;
legacy markers are ignored. A fresh child needs one authorized parent lookup,
reused during configuration preparation. Roots and recovery with valid owned
progress need no parent lookup; a child without progress re-reads its parent.
Required graph reads and write confirmation remain, but there is no separate
recovery-validation read or branch-origin, boundary, or global-head record write.

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

After configuration hooks and saver setup, validate that saved root or child
progress belongs to the current response key and requested parent, if any.
Matching the thread alone is insufficient. Ownership is checked in the graph's
checkpoint read, without a separate preflight read. Branching requires a matching
stamp; ordinary recovery also rejects mismatched stamps but accepts absent
stamps for compatibility with pre-adapter snapshots.

Valid progress needs no parent lookup. A child without progress re-validates its
stored parent; a root without progress replays its original request. Target
conversation recovery must instead reuse the admitted starting reference,
not freshly select a mutable conversation head.

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
- [Branch preparation](../../langchain_azure_ai/agents/hosting/_responses/branching.py):
  admission, parent selection, and eligibility after configuration hooks.
- [Task storage and checkpoint adaptation](../../langchain_azure_ai/agents/hosting/_responses/task_storage_manager.py):
  shared progress references, exact reads, ownership, persistence barriers, and
  native recovery heads.
- [Graph-stream conversion](../../langchain_azure_ai/agents/hosting/_converters/_stream.py):
  graph persistence before SDK progress publication.
- [Branching tests](../../tests/unit_tests/agents/hosting/test_response_branching.py):
  existing parent preservation, concurrent sibling, failure, and recovery coverage.
  The target contract below requires additional compatibility coverage.

The branching helpers and saver wrapper are internal, not public extension points.

## Implementation gaps and verification

This revision changes the specification only. Implementation gaps and unverified
behavior are distinguished below; neither indicates completed support.

| Topic                              | Target contract                                                                                                                                            | Implementation status                                                                                                                                        |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Explicit response storage provider | Branching requires an explicitly supplied provider.                                                                                                        | **Implementation required:** The constructor selects a default provider when `store` is omitted.                                                             |
| Conversation lookup                | Reject missing resources before graph execution; distinguish them from existing empty conversations.                                                       | **Implementation required:** Local providers do not enforce this distinction; see provider differences below.                                                |
| Conversation context and linkage   | Use shared checkpoint execution; write back only the completed, retained turn's final checkpoint, preserving public linkage fields and branch isolation.   | **Implementation required:** Explicit conversations use ordinary continuation and HITL routing; the existing pointer can be updated before finalization.     |
| Request `store=False`              | Preserve the retained conversation head and prior graph state; clean up finalized execution-owned checkpoints on every continuation and finalization path. | **Implementation required:** Ordinary continuation updates its checkpoint pointer without checking request `store`; isolation and cleanup must be completed. |
| Request-scoped `instructions`      | Do not inherit the parent's top-level instructions through its checkpoint.                                                                                 | **Verification required:** Check checkpoint restoration against this rule.                                                                                   |

**Future scope:** Pending-parent tool and approval branching is not part of this
scope; the current branching path rejects those replies. See
[Future scope](#future-scope-function-calls-and-mcp-approvals).
Full input replay for unretained responses remains an application responsibility,
not automatic host support; see
[Function-call outputs and MCP approval replies](#function-call-outputs-and-mcp-approval-replies).

**Provider differences:** `InMemoryResponseProvider` and `FileResponseStore`
return empty history for both missing and existing empty conversations. Without
a saved checkpoint, the adapter can use the ID as the graph thread and execute
from empty state. Enabling branching does not add an existence check. With
`FoundryStorageProvider`, lookup is delegated to the remote service; in the
inspected Agent Server SDK 2.3.0, it occurs before the graph handler and surfaces
backend not-found errors. Check custom providers separately; local first-use
acceptance is not a promised extension to the target contract.

**Service observations:** Direct Azure Responses REST requests with an unknown,
valid-format conversation ID returned HTTP 404 with both `store=True` and
`store=False`. Exact error details at OpenAI's public endpoint remain unverified.

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
  advances. During a multi-step graph run, the head must remain at the prior
  final checkpoint. Only successful finalization advances it to that run's exact
  final checkpoint; failure, cancellation, or terminal response-storage failure
  must not advance it.
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
