# Response Branching

Response branching lets you send new input using the context of an earlier
response, called the **parent**. Use it to continue from an answer, try a
different direction, or regenerate an answer.

Checkpoint-based branching is available through `ResponsesHostServer` and is
disabled by default.
**It requires a stored parent and its exact graph checkpoint.**
The server uses a **response storage provider** for response records, separate
from the graph checkpointer. Branching accepts the host's default provider:
`FileResponseStore` locally or `FoundryStorageProvider` on Foundry. Passing
server `store=provider` optionally overrides that default.

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
  not a response ID. Lookup and parent selection follow ordinary continuation.
- `previous_response_id` and `conversation` are mutually exclusive.
- Request `store` controls response retention and defaults to `True`. Enabling
  branching must not silently override `store=False`. See
  [Request store behavior](#request-store-behavior) for response retention.
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

Ordinary continuation owns request admission, parent selection, and persistence
of each response's checkpoint reference. Branching reuses that contract: once a
stored, eligible parent is selected, it restores that parent's exact checkpoint
and creates an independent child.

A stored response created within a conversation can be selected using
`previous_response_id` when the common continuation rules allow it. The branch
restores that response's persisted exact checkpoint, not a mutable conversation
head. Cross-linkage and conversation-head policy are separate work tracked under
[TODO](#todo).

## Quick start

### 1. Enable branching on the server

Starting from the [basic hosting setup](../../README.md#hosting-langgraph-agents-on-microsoft-foundry)
with a configured `model`, enable branching on a graph with a checkpointer:

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_azure_ai.agents.hosting import ResponsesHostServer

graph = create_agent(model, tools=[], checkpointer=InMemorySaver())

if __name__ == "__main__":
    ResponsesHostServer(
        graph,
        enable_response_branching=True,
    ).run(port=8088)
```

  Server `store=provider` is an optional response-provider override, not a boolean.
  `InMemorySaver` is for local demos; use a durable, history-preserving saver for
  reuse after restart.

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

The parent eligibility rules below cover only branching from finished parent
graphs. Ordinary tool and approval continuation keeps its existing behavior.
Branching from a paused parent is future work, including isolation of tool replies,
approval decisions, and resume state between children of that parent. Supplying
matching replies does not make a paused parent eligible for this branching path.

A response-ID parent is reusable only when:

1. It was stored with `status="completed"` and is available to the requesting caller.
2. Its stored internal metadata identifies its exact final graph checkpoint.
3. That checkpoint and the history needed to restore it are still available.
4. Its graph has finished, with no pending tasks or interrupts, and its output
   has no outstanding function calls or MCP approval requests.

Parents created before branching was enabled or through ordinary continuation,
including responses saved in a conversation, are eligible when the common
continuation rules admit them and they satisfy these checks. No branching-specific
marker is required. Later responses do not change the checkpoint selected by an
earlier response ID.

**Response completion and graph completion are different.** A response can
contain tool calls or approval requests while its graph is paused. Such a parent
is outside the current branching scope, even if its response status is
`completed`. Current branching restores a finished checkpoint.

### Instructions, checkpoints, and routing

Do not inherit a previous response's top-level `instructions` through a graph
checkpoint. Distinguish request-scoped instructions from actual system or
developer message items supplied in conversation history.

The graph's saver must preserve checkpoint history and implement asynchronous
reads and writes. Keep referenced checkpoints and any history needed to restore
them, not just the latest checkpoint.

With `enable_response_branching=True`, request routing reuses ordinary
continuation's admission and linkage rules:

| Request linkage                   | Behavior                                                                                      |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| Neither parent nor `conversation` | Start a new root response in a new thread.                                                    |
| `previous_response_id` only       | Start from the parent's exact checkpoint.                                                     |
| `conversation` only               | Use ordinary continuation; conversation-head policy is outside this feature.                  |
| Both parent and `conversation`    | Reject with HTTP 400.                                                                         |

Omitted or `null` linkage fields do not select a parent or conversation.
Resolve references within the request's authorized storage scope; a resource ID
from another scope is not sufficient. Use the common continuation rules for
reference errors rather than substituting another parent.

Keep `options.steerable_conversations=False` and `app=None`. Steering hands off
an in-flight turn; checkpoint-based branching selects a saved response boundary.
The adapter must create its own host to configure branching. Unsupported
combinations, or a saver without the required async methods, raise `ValueError`
during initialization.

The branching flag controls the checkpoint execution strategy for supported
requests. It must not change the public meanings of `conversation`, `store`,
or `instructions`.
The constructor's existing automatic provider selection remains available with
branching enabled or disabled; `store=provider` is an optional override.

## Request `store` behavior

Request `store` is a boolean and defaults to `True`. It controls retention of the
current response, independently of the server's provider configuration. Enabling
branching must honor this choice rather than force response retention.

A response created with `store=False` is not retained and cannot later be selected
as a response-ID parent. A branch requires a stored, accessible parent, regardless
of whether the child itself uses `store=True` or `store=False`. The child's
retention choice must not delete or modify its stored parent.

The effects of `conversation` with `store=False` on item appends and head
advancement are a separate Responses/Conversations topic, including Azure
observations and OpenAI compatibility verification; see [TODO](#todo).
Conversation-state isolation,
temporary checkpoint cleanup, and full-input replay responsibilities also belong
to a separate execution-lifecycle design. This branching feature does not define
those behaviors or add them as implementation or verification requirements.

## What must be stored?

History-based continuation uses SDK-managed response and history storage.
Checkpoint-based branching additionally needs graph checkpoints:

| Data                                                               | Where it lives             | How to configure it                                |
| ------------------------------------------------------------------ | -------------------------- | -------------------------------------------------- |
| Response, status, input/output, and internal checkpoint references | SDK response provider      | Host default; optional `store=provider`            |
| Graph state, pending writes, and checkpoint history                | LangGraph checkpoint saver | Graph: `checkpointer=...`                          |

The server uses its default response provider unless `store=provider` overrides
it. The client's `store` is a boolean that defaults to `True` and controls
retention of that response. Having a provider does not force every response to
be retained.

Response retention does not replace the graph's checkpointer. The application
owns the graph saver's initialization, lifecycle, and general retention policy.

**Migration:** Existing automatic provider selection remains valid when enabling
branching; no explicit provider argument is required. If overriding the provider,
preserve its backend configuration and existing data. Selecting a different
provider does not migrate response records or graph checkpoints.

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
policy.

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
parents awaiting function results or MCP approval decisions is future work;
ordinary tool and approval continuation keeps its existing behavior.

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

1. **Select.** Reuse ordinary continuation's request admission and parent
  selection. For a supported finished parent, load its stored exact checkpoint
  reference through `select_branch_checkpoint` within the platform's
  authorization context. `branch_selection_scope` caches this selection only
  for the request. A child without saved progress re-reads its stored parent.
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
  final checkpoint reference; no separate boundary is written.

Graph writes and SDK response storage are **not a transaction**. A graph
checkpoint alone cannot make a reusable parent: its response must also be stored
and completed, and its output and restored graph state must satisfy the current
finished-parent requirements. A response whose graph pauses for a tool result or
approval is not a reusable parent within this scope.

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
stored parent; a root without progress replays its original request.

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
| Response storage provider          | Accept the host's default provider or an explicit `store=provider` override.                                                                                 | Automatic provider selection already exists and must remain available when branching is enabled.                                                            |
| Request-scoped `instructions`      | Do not inherit the parent's top-level instructions through its checkpoint.                                                                                 | **Verification required:** Check checkpoint restoration against this rule.                                                                                   |

**Future scope:** Branching from paused parent graphs, including isolation of
tool replies, approval decisions, and resume state between children, is deferred.
Ordinary tool and approval continuation keeps its existing behavior.

Before claiming compatibility, cover these observable scenarios:

- The host's default provider and a history-preserving saver enable branching.
  Cover local `FileResponseStore`, Foundry `FoundryStorageProvider`, and an
  explicit provider override. Omitting `store` must not fail initialization;
  boolean constructor values are not response providers.
- B and C can select A independently, including when a newer sibling exists.
  A child with `store=False` does not delete or modify its stored parent.
- A stored, eligible response created in a conversation can be selected using
  only `previous_response_id` when ordinary continuation admits that parent,
  even after newer responses exist. Its branch uses the selected response's exact
  checkpoint, not the conversation's latest state. Conversation-head effects
  are not part of this verification.
- Return matched function outputs and approved or denied MCP decisions through
  the same linkage used for ordinary continuation.
- Verify HTTP errors, error objects, terminal SSE events, and absence of graph
  execution on caller-input failures, not only internal helper results.

The cited documentation and service verification do not establish every edge
case. Confirm these remaining cases against the service before claiming
compatibility:

- Exact status, error code, and parameter attribution for reference errors not
  already specified above, including inaccessible resources and malformed
  response references, reusing the common continuation rules.
- Acceptance of parents in different response statuses.

These are compatibility verification items, not permission to invent stricter
branching-only rules or silently fall back to a different conversation or parent.

## TODO

Discuss conversation linkage and pointer movement as a separate ordinary
continuation design topic, not as branching implementation or verification
requirements:

- Define the shared continuation rules for request admission, parent selection,
  and persistence of each response's exact checkpoint reference. Branching reuses
  those rules within its supported finished-parent scope.
- Resolve the cross-linkage case: A is stored in an explicit conversation, B uses
  `previous_response_id=A.id`, and C later uses the original `conversation` ID.
  Decide whether B advances that conversation's pointer and which parent and
  context C selects, without assuming that B is included or excluded.
- Review pointer publication. The existing pointer uses the user-scoped SDK
  `conversation_chain_id`, not a global LangGraph thread head. Ordinary
  continuation can update it after streaming, before shutdown/cancellation
  checks and terminal response persistence. Decide the intended behavior for
  intermediate progress, pauses, failures, cancellation, recovery, and concurrent
  turns, including whether recovery reuses an admitted starting reference.
- Define conversation lookup behavior across providers. Local memory/file
  providers currently return empty history for both missing and existing empty
  conversations, while the inspected Foundry-backed path delegates lookup to the
  service. Resolve existence and authorization handling in the common
  continuation contract, rather than adding branching-only rules.
- Define `conversation` with `store=False` item-append and head-advancement
  behavior, including Azure observations and OpenAI compatibility verification.

If the common continuation rules allow a request to select B, branching restores
B's persisted exact checkpoint reference and creates an independent child.
Otherwise, the request is rejected under those rules; branching does not decide
which parent a conversation request selects.
