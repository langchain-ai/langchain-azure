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

Except for the conversation create-on-first-use rule specified here, public HTTP
request and response behavior must follow the
[OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)
and [conversation-state guide](https://developers.openai.com/api/docs/guides/conversation-state).
The [OpenAI quickstart](https://developers.openai.com/api/docs/quickstart?tool-type=web-search)
provides the client usage baseline; the API reference defines the field semantics.
This does not add OpenAI-hosted models or built-in tools to the configured graph.

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
- A valid, previously unused `conversation` ID creates an empty conversation on
  first use. An existing ID continues that conversation's current state. This
  creation rule is a local contract; its OpenAI equivalence is not yet verified.
- `previous_response_id` and `conversation` are mutually exclusive.
- Request `store` controls response retention and defaults to `True`. Enabling
  branching must not silently override `store=False`.
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

| Reference | Starting state |
| --- | --- |
| `previous_response_id=A.id` | A's exact saved state, regardless of later responses. |
| `conversation=conversation.id` | The conversation's current state; in a sequential exchange, normally its latest response's checkpoint. |

If a conversation advances through A, B, and D, its reference moves forward to D.
Referencing A directly still starts from A. Both linkage forms can use the same
internal state-selection model.

This abstraction preserves the public API: a conversation remains a separate
resource, and the linkage fields remain mutually exclusive. A valid new
conversation ID creates an empty conversation and starts fresh. Initial items
and independently added or deleted items must be reflected in the selected state;
the latest response's checkpoint is usable only when it represents that
conversation context.

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

### 4. Create or continue a conversation

Supply a new conversation ID to create an empty conversation on first use, then
reuse that ID on successive requests:

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

The string and object forms of `conversation` refer to the same resource.
Conversation items precede the new input; the response's input and output items
are added to that conversation after it completes. In a sequential exchange,
the next turn therefore includes the preceding turn.

A valid new ID creates a conversation with no history and starts a root response
in a new thread. An existing ID continues that conversation's current state.
Explicit creation through `client.conversations.create()` also remains supported,
including conversations with initial items before their first response.

## Requirements and request rules

> **Current scope:** Branching requires a finished parent graph with no pending
> tasks, interrupts, function calls, or MCP approval requests. Parents with
> outstanding `function_call` or `mcp_approval_request` items remain ineligible
> even when matching replies are supplied. Support for these branching flows is
> deferred to the future scope below. Ordinary tool and approval continuation
> behavior is unchanged.

A response-ID parent is reusable only when:

1. It was stored and is available to the requesting caller.
2. Its stored internal metadata identifies its exact final graph checkpoint.
3. That checkpoint and the history needed to restore it are still available.
4. Its graph has finished, with no pending tasks or interrupts, and its output
   has no outstanding function calls or MCP approval requests.

Parents created before branching was enabled or through ordinary continuation
are eligible when they satisfy these checks. No branching-specific marker is
required.

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

With `enable_response_branching=True`, request routing is:

| Request linkage | Behavior |
| --- | --- |
| Neither parent nor `conversation` | Start a new root response in a new thread. |
| `previous_response_id` only | Start from the parent's exact checkpoint. |
| Existing `conversation`, without a previous response | Start from its existing items and the new input. An empty conversation starts a fresh root. |
| Existing `conversation`, with previous responses | Continue its ordered item context. For sequential turns, use the latest response's checkpoint when it represents that context. |
| Valid, previously unused `conversation` | Create an empty conversation with the supplied ID and start a root response in a new thread. |
| Both parent and `conversation` | Reject with HTTP 400. |

Omitted or `null` linkage fields do not select a parent or conversation.
Creation applies only to valid IDs confirmed absent in the request's authorized
storage scope. Authorization errors and storage failures must not trigger
conversation creation.
The checkpoint shortcut must account for initial conversation items and any
independently added or deleted items. A latest-response pointer alone is not the
conversation's public history. Internal parent resolution must not rewrite the
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

## What must be stored?

Branching needs **both** response records and graph checkpoints:

| Data | Where it lives | How to configure it |
| --- | --- | --- |
| Response, status, input/output, and internal checkpoint references | SDK response provider | Server: `ResponsesHostServer(..., store=provider)` |
| Conversation resource and ordered items | SDK conversation storage | Create on first use of a valid new ID, or create explicitly through the Conversations API. |
| Graph state, pending writes, and checkpoint history | LangGraph checkpoint saver | Graph: `checkpointer=...` |

With branching enabled, the server's `store` is a required explicit provider;
it configures storage capability and the backend. The client's `store` is a
boolean that defaults to `True` and controls retention of that response. These
are separate contracts: having a provider does not force every response to be
retained, and a client setting `store=True` cannot supply a missing provider.

Use an appropriate provider explicitly, such as `FileResponseStore` for a local
demo or `FoundryStorageProvider` on Foundry. Neither response retention nor
conversation-item persistence replaces the graph's checkpointer. Disabling
response retention must not be treated as deleting a conversation's items.
The application owns the graph saver's initialization, lifecycle, and retention.

**Migration:** Branching callers that relied on automatic provider selection
must pass that provider explicitly. Preserve its backend configuration and
existing data; selecting a different provider does not migrate response records
or graph checkpoints.

**Retaining a response does not retain or renew its graph checkpoints.**
Durable backends can still expire data. Align both lifetimes, retain history
needed by recoverable tasks, and clean up unneeded data through the backends.
The host does not coordinate retention or cascade response deletion to checkpoints.
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

| Problem | Result and action |
| --- | --- |
| Request includes `response_id`, an invalid parent ID, or both linkage fields | HTTP 400 before execution. Omit `response_id`; use a nonempty string or `null` for the parent, and choose one linkage type. |
| Referenced response is missing or inaccessible | Return the corresponding OpenAI-compatible reference error before graph execution. Do not substitute a new root response. |
| Conversation ID is malformed, or lookup fails due to authorization or storage errors | Return the appropriate error before graph execution. Only confirmed absence of a valid ID permits creation; lookup failures must not create a replacement conversation. |
| Stored parent lacks a checkpoint reference or its graph state cannot be restored | Backend failure. Restore the required data or choose another parent. |
| Input contains approval or tool-result replies | Validate the linked calls and use the appropriate continuation path; these input types are not blanket errors. |
| Exact checkpoint is missing or mismatched | Failed response. Restore the required data or choose another parent; there is no fallback to the latest checkpoint. |
| Graph checkpoint write fails | Failed response. Progress is not published for the missing checkpoint; investigate the saver backend. |
| SDK terminal response storage fails | The graph checkpoint alone cannot serve as a parent. Check SDK storage errors and the stored response status. |

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

| Saved state | Recovery behavior |
| --- | --- |
| Root response, with or without saved progress | Fail; submit a new root request. |
| Child with no saved progress | Load and validate the parent's exact checkpoint, then replay the child's input. Fail if the parent or required checkpoint is unavailable. |
| Child with valid, response-owned progress | Resume that checkpoint with input `None`, preserving pending writes. No parent response lookup is needed. |
| Child's saved graph is already finished | Finish response finalization without running graph nodes again. |
| Malformed progress, unavailable/mismatched checkpoint, or wrong ownership | Fail; do not fall back to the parent or the thread's latest checkpoint. |

Recovery uses the last response snapshot the SDK actually persisted. If graph
execution advanced beyond that snapshot, work and external effects can repeat.
Even with a finished graph checkpoint, response finalization and storage may run
again. Keep the checkpoint history needed to restore the child's state.

## Implementation notes

The following details are for maintainers; applications do not need to manage
these fields or call these helpers.

<details>
<summary>Checkpoint selection, persistence, and recovery</summary>

### Checkpoint references

| Name | Meaning |
| --- | --- |
| Origin | The parent's exact checkpoint selected before a child starts. No separate origin record is stored. |
| Progress | This response's latest confirmed graph checkpoint, referenced by SDK recovery metadata. |
| Final checkpoint | The last progress reference stored with a completed response. Parent eligibility is checked against the restored graph state and response output. |

These are roles for references, not copies of graph state or three separate
records. A root response has no parent; both root and child responses use the
root checkpoint namespace.

### Fresh request

1. **Select.** Resolve the explicit parent or the conversation context using the
   platform's authorization context. Create an empty conversation on first use
   of a valid ID confirmed absent in that scope. Response-ID selection uses
   `select_branch_checkpoint`; `branch_selection_scope` caches it only for this
   request. Persist the resolved linkage for recovery before graph execution.
2. **Validate.** After configuration hooks and graph-bound configuration merging,
   re-pin the parent's thread, root namespace, and checkpoint ID. Read through
   `validate_branch_checkpoint` and `ResponseCheckpointSaver`. The returned tuple
   and payload must match the selected identity. A finished-answer branch has no
   next tasks or interrupts; tool and approval continuation must instead validate
   its replies against the selected pending state. Validation failure prevents
   execution. Extending the current finished-parent helpers for that continuation
   is an implementation requirement, not existing support.
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
   emit `response.completed` for SDK terminal storage. The response retains the
   same confirmed final checkpoint reference; no separate boundary is written.

Graph writes and SDK response storage are **not a transaction**. A graph
checkpoint alone cannot make a reusable parent: the stored response must also
pass the graph-state and input checks for the selected continuation path.
Response-ID branching must not select a mutable conversation head.
Conversation continuation may use `conversation_chain_store` to locate the
latest response's checkpoint, but that pointer must remain consistent with the
SDK's ordered conversation items. It is not a replacement for the conversation
resource. That store defaults to `FoundryConversationChainStore`.

### Internal metadata and isolation

- SDK progress fields: `langgraph_thread_id` and `langgraph_checkpoint_id`.
- Admitted mode: `langgraph_response_branching="checkpoint-v1"`. Recovery uses
  the original mode, not the current server flag. Middleware strips client mode
  headers; inconsistent recovery mode fails.
- Graph ownership: `langgraph_response_execution` stores the JSON-encoded array
  `[response_key, parent_response_id]`; the parent is `null` for a root response.
  Private subgraph checkpoints are not stamped.
- A conversation's internally resolved parent or item snapshot must be retained
  separately from the public request fields. Recovery must use that admitted
  context, not whatever response is latest when recovery starts.
- A root thread uses the response key: the response ID prefixed with
  `user-<SHA-256 hex digest of user_id_key>:` when a nonempty platform
  `user_id_key` is present. This is separate from backend `user_isolation`.

### Recovery details

After configuration hooks and saver setup, validate that saved child progress
belongs to both the current response key and the parent resolved for that request.
Matching the thread alone is insufficient. Valid progress needs no parent lookup;
absent progress requires the same admitted parent or conversation snapshot,
not a fresh selection of a mutable conversation head.

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

</details>

## Implementation gaps and verification

This revision changes the specification only. In particular:

- The current constructor selects a default provider when branching is enabled
  and `store` is omitted. It must be changed to require an explicit provider.
- The current branching path supports finished parents and rejects approval and
  tool-result replies. Supporting the documented OpenAI continuation flows
  requires implementation work.
- Explicit conversations currently use the ordinary continuation path. Verify
  resource creation, ordered items, checkpoint selection, and public response
  fields against the conversation contract above.
- Checkpoint restoration must be verified not to inherit request-scoped
  `instructions` from the parent.

Before claiming compatibility, cover these observable scenarios:

- A valid explicit provider and a history-preserving saver enable branching;
  missing providers and boolean constructor values are rejected.
- B and C can select A independently, including when a newer sibling exists.
  A child with `store=False` does not delete or modify its stored parent.
- Create a conversation on first use of a valid new ID, then reuse it with both
  ID forms and verify continuation from the same conversation's current state.
  Also cover explicit creation through the Conversations API, initial items,
  independently modified items, and recovery after the conversation advances.
- Malformed conversation IDs, authorization errors, and storage lookup failures
  must not create conversations or start graph execution.
- Return matched function outputs and approved or denied MCP decisions through
  the same linkage used for ordinary continuation.
- Verify HTTP errors, error objects, terminal SSE events, and absence of graph
  execution on caller-input failures, not only internal helper results.

The cited documentation does not establish every edge case. Confirm the following
against OpenAI's HTTP API before claiming compatibility for those behaviors:

- OpenAI's handling of a valid new conversation ID supplied directly, compared
  with this spec's explicit create-on-first-use rule.
- Exact status, error code, and parameter attribution for missing responses,
  inaccessible resources, and malformed response or conversation references.
- Acceptance of parents in different response statuses.
- Partial, missing, duplicate, or unknown tool outputs and approval decisions,
  including requests with multiple outstanding calls. Do not assume function
  outputs and MCP approvals have identical batch-completeness rules.
- Concurrent requests to one conversation and the visibility/order of its items.
- Response retrieval and conversation-item retention with
  `conversation` combined with `store=False`.

These are compatibility verification items, not permission to invent stricter
branching-only rules or silently fall back to a different conversation or parent.
