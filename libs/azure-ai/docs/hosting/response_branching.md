# Response Branching in Hosting

## Protocol Scope

`langchain-azure-ai` supports both `ResponsesHostServer` and
`InvocationsHostServer` as hosting protocols. Response branching over
HTTP is available only through Responses with `enable_response_branching=True`.
The option defaults to `False`. Its checkpoint-selection guarantee is
**root-only** (`checkpoint_ns=""`); private subgraph state remains
application-owned, as described in [Root-Only State and Subgraphs](#root-only-state-and-subgraphs).

Invocations supports checkpointed session continuation and configured task
recovery, not user-selected historical checkpoints. Its `previous_invocation_id`
is an ordering precondition against the latest accepted turn, not a checkpoint
selector. Native LangGraph APIs on `host.graph` are separate from this HTTP
contract.

## Responses Behavior

For checkpointed graphs, fresh requests follow the paths below. Conversation
and legacy chains use their current saved checkpoint, which can advance as new
responses complete. In a legacy chain, referencing an earlier response ID does
not necessarily restore that response's state.

| Configuration and linkage | Starting checkpoint/state | Behavior |
| --- | --- | --- |
| Branching enabled, `previous_response_id`, no `conversation` | The exact root checkpoint recorded for the authorized, stored, completed parent response | Continue, fork, or regenerate in the parent's thread without changing its root checkpoint or pending writes. Private subgraph state remains application-owned. A paused parent requires matching resume/approval input; otherwise return its pending interrupt without running the graph. |
| Branching enabled, neither `previous_response_id` nor `conversation` supplied | A new response-scoped thread with no prior checkpoint | Start an independent root. Omitting a linkage field or setting it to `null` has the same effect. |
| Either branching setting, explicit `conversation`, no `previous_response_id` | The conversation's saved checkpoint pointer, or the resolved conversation thread's latest checkpoint when no pointer exists | Use legacy conversation continuation, not response-boundary branching. A new conversation starts without saved state. |
| Branching disabled, `previous_response_id`, no `conversation` | The legacy chain's saved checkpoint pointer, or the ancestry-resolved thread's latest checkpoint when no pointer exists | Continue legacy graph state; the parent ID does not guarantee restoration of that response's historical checkpoint. |
| Branching disabled, neither `previous_response_id` nor `conversation` supplied | A new response-scoped thread with no prior checkpoint | Start a new root using the legacy path. |
| Branching enabled, both linkage fields non-null | None | Reject before execution; neither linkage field takes precedence. |
| Branching disabled, both linkage fields non-null | Determined by the existing SDK and legacy linkage handling | No additional adapter validation; existing SDK behavior is unchanged. Clients should choose only one linkage field. |

Without a graph checkpointer, `enable_response_branching` must remain `False`.
Responses restores only message history, not other graph state such as counters
or application variables.

## Meaning of Response Branching

Response branching backed by LangGraph checkpoints follows the
[OpenAI Responses parent-selection model][openai-create]: `previous_response_id`
selects a response boundary, not an arbitrary or intermediate graph checkpoint.
The API does not expose checkpoint selection or state editing.

If B was created from A, another request selecting A starts from A's root state,
not B's or the thread's latest root state. Selecting B continues after B; it
does not rerun B. Restored root state includes non-message values, subgraph
outputs, and application state references saved in the root. Send only new
input, not a duplicate transcript of the selected response's history.

### Root-Only State and Subgraphs

The host selects and preserves root checkpoints. It does not select, copy,
rewind, or isolate private subgraph checkpoints, even when a subgraph uses the
same saver and thread as the root. Root and subgraph namespaces have separate
checkpoint histories; selecting a root checkpoint is not a whole-graph rollback.

| Host responsibility | Application responsibility |
| --- | --- |
| Restore the selected response's exact root checkpoint and all state saved in it. | Choose the state or checkpoint from which each subgraph runs. |
| Restore subgraph outputs and snapshot references already stored in root state. | Manage private subgraph snapshots, branching, concurrency, retention, and access isolation. |
| Preserve the selected root checkpoint when creating another response. | Decide whether private subgraph memory is branch-local or intentionally shared. |

Persistent subgraphs, including `compile(checkpointer=True)`, are not rejected
merely because they retain memory across calls. Without application-managed
selection, a persistent subgraph can read its latest state, including work from
a sibling branch. For example, a private counter can advance A=1, B(A)=2,
C(A)=3 even though C correctly restores A's root state. Applications requiring
C(A)=2 must manage the child's state. The host does not change the subgraph's
checkpointer setting or promise automatic subgraph isolation.

Default per-invocation subgraphs (`compile()` or `checkpointer=None`) keep
LangGraph's ordinary behavior: each invocation starts from its supplied input,
without automatically inheriting the previous invocation's private memory.
Subgraph output that is written back into root state is still covered by root
checkpoint restoration.

One application-managed pattern is:

1. Save a reference such as `child_state_ref` in the root state.
2. In the node that calls the child, load the immutable snapshot identified by
   that reference and supply its state to a per-invocation subgraph.
3. After successful execution, save a new snapshot under a new reference and
   return that reference to the root. Do not mutate the parent's snapshot.
4. Initialize fresh child state only when the root has no reference. A missing
   or unreadable referenced snapshot must fail explicitly, not reset to empty
   state or fall back to latest.

Restoring A's root state restores A's reference, but loading and applying it is
application code, not automatic host behavior. An in-memory snapshot store is
suitable only for local single-worker use; durable deployments need application
storage with compatible retention and user isolation.

This responsibility split does not remove the existing restriction on
[forking paused subgraphs](#recovery-and-approvals).

### Thread and Checkpoint Identity

New continuation, fork, and regeneration requests retain the selected parent's
LangGraph `thread_id`. Independently created roots still get separate threads.
Each response selects an exact root checkpoint; a shared thread's latest root
checkpoint does not select a branch. Private subgraph state follows the
application's policy above.

For a completed graph, execution starts directly from the parent's root checkpoint.
For a paused graph, the host creates a new checkpoint in the same thread and root
namespace (`checkpoint_ns=""`) before resuming. Root pending writes are transferred
to the fork's task IDs, preserving already answered sequential or parallel work.
The client's original `call_id` or `approval_request_id` is mapped to the fork's
interrupt ID, so clients do not need to discover new internal IDs before replying.
Both siblings can answer the same parent's pause independently.

Waiting and partial-approval responses retain the wire IDs of unchanged,
unanswered approvals. Their mappings and original approval payloads are stored
in the branch store, not inline in response metadata, whose value has a
512-character limit. This preserves reply IDs across partial and waiting turns.

Sibling requests can overlap in the supported single-worker setup. If A is an
eligible stored, completed parent, C can start from A while sibling B is still
`in_progress`. C is not queued or rejected merely because B is running, and
starts from A's root checkpoint, not B's root progress. Selecting the still-running B
itself as the parent fails. SDK admission and resource limits still apply.

Regenerating B submits its original input with A as `previous_response_id` and
receives a new response ID. Identical output, idempotent retry, and exactly-once
tool effects are not guaranteed. SSE replay and recovery of the same task are
separate operations, not new historical branches.

## JSON and SSE Output

Each branch has a new response and, with `stream=true`, a new SSE stream; it
does not replay its parent's events. These examples omit unrelated fields and
abbreviate IDs. Use actual returned IDs. Output text depends on the application graph.

### Normal Completion

With A as the selected parent, input B produces a new response:

```json
{
  "id": "resp_B",
  "previous_response_id": "resp_A",
  "status": "completed",
  "output": [
    {
      "type": "message", "id": "msg_B", "role": "assistant",
      "content": [{"type": "output_text", "text": "A,B"}]
    }
  ]
}
```

With `stream=true`, the example's event sequence is:

```text
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta
response.output_text.done
response.content_part.done
response.output_item.done
response.completed
```

Lifecycle SSE payloads put the response object under `response`. At creation
and in-progress it has `id="resp_B"`, `previous_response_id="resp_A"`,
`status="in_progress"`, and `output=[]`; at completion its fields match the
JSON example. Continue with `previous_response_id="resp_B"`, or select
`resp_A` for a sibling. Message IDs and event sequence numbers are not parent IDs.

### Interrupt and Resume

Branching selects the starting checkpoint without changing LangGraph interrupt
semantics. If a branch reaches an interrupt, it pauses and requires fresh,
matching resume or approval input for that branch, even if a sibling branch has
already answered the same question. Results already captured before the selected
checkpoint remain part of the restored state and are not re-executed merely
because a new branch is created.

For parallel approvals A and B, approving A produces a new paused response
R_A. B retains its `mcp_approval_request.id`, `arguments`, and paired
`function_call.call_id`. Continue with `previous_response_id=R_A.id` and B's
approval ID. The host reuses A's persisted task output rather than executing A again,
even while B keeps the same graph step unfinished. An unfinished node still
follows LangGraph's normal interrupt behavior and restarts from its beginning
when resumed. Selecting the original response instead creates an independent
branch; it does not accumulate the approvals submitted to R_A.

```text
P:  A(mcpr_a), B(mcpr_b)
|-- approve A --> R_A: B(mcpr_b)
|                  `-- approve B --> R_AB: complete
`-- approve A --> S_A: B(mcpr_b), independently pending
```

Approval routing is scoped to the selected response, not globally to an mcpr ID.
Answering B in R_A does not answer it in S_A. To continue R_A, select R_A even
when reusing an approval ID originally received from P.

Reused MCP items keep their original payload, including the creating
`response_id`, because SDK stores may index items globally by ID. Always use the
containing response's `id`, not an approval item's `response_id`, as the parent
for the next continuation. Function-call item IDs can still change; their
`call_id` identifies the outstanding approval.

A changed approval payload or a new interrupt after an answered approval gets a
new identity, even in the same node with identical prompt text. An update-only
command does not count as answering an interrupt. Stale answered IDs cannot
approve the next stage.

Normal response chains inherit the SDK storage partition and retain the full
mcpr ID. If a custom response-ID override changes that partition, the host emits
a partition-compatible mcpr ID instead of reusing an incompatible storage ID.
The wire interrupt ID and branch-local reply routing remain intact.

If an original item's `agent_reference` was null and the new response supplies
one, SDK item defaults would change the original payload. The host gives the
outstanding approval a new identity instead of overwriting the historical item.

For a graph that asks for a name through `interrupt()`, a root response P can
finish with these pending output items:

```json
{
  "id": "resp_P",
  "previous_response_id": null,
  "status": "completed",
  "output": [
    {
      "type": "function_call", "id": "fc_P", "call_id": "pause_1",
      "name": "__hosted_agent_adapter_interrupt__",
      "arguments": "{\"interrupt_id\":\"pause_1\",\"value\":\"name?\"}",
      "status": "completed"
    },
    {
      "type": "mcp_approval_request", "id": "mcpr_P", "server_label": "langgraph",
      "name": "__hosted_agent_adapter_interrupt__",
      "arguments": "{\"interrupt_id\":\"pause_1\",\"value\":\"name?\"}"
    }
  ]
}
```

The corresponding SSE sequence emits the pending items before completion:

```text
response.created
response.in_progress
response.output_item.added                 (function_call)
response.function_call_arguments.delta
response.function_call_arguments.done
response.output_item.done                  (function_call)
response.output_item.added                 (mcp_approval_request)
response.output_item.done                  (mcp_approval_request)
response.completed
```

The terminal event's `response` matches P's JSON above: the response and stream
end, but the graph remains paused. `function_call` and `mcp_approval_request`
represent the same pause through `arguments.interrupt_id`, not two approvals.
Their item IDs differ. A completed function call item does not mean the user
has answered.

If a request selects paused P and sends only ordinary text, such as
`input="Alice"`, with no matching resume or approval input, the host returns the
pending interrupt again in a new `completed` response. It does not run the graph,
process that text as an answer, or add it to the graph state. The same waiting
behavior applies to other input that contains no matching resume/approval item.

Resume this custom interrupt with a new request selecting P:

```json
{
  "model": "test", "previous_response_id": "resp_P", "store": true,
  "input": [
    {"type": "function_call_output", "call_id": "pause_1", "output": "Alice"}
  ]
}
```

This starts a new response R, not a replay of P, and a new stream if requested.
R has `previous_response_id="resp_P"`; use R's `id` for the next continuation.
MCP approval decisions use
`mcp_approval_response` with the returned approval request's `id` as
`approval_request_id`; function-call results use `call_id` as shown above.

With branching enabled, resuming P continues from its saved pause in an isolated
checkpoint copy, leaving P unchanged. By contrast, selecting an earlier response
boundary with new input starts an independent branch from that state rather than
answering P's interrupt. Both paths create a new response and follow the same
interrupt and resume rules.

After a partial approval, continue from the latest response's `id`. In an
interrupt-ID-keyed `resume` map, current IDs are remapped independently; stale IDs
do not prevent valid decisions from being applied.

Checkpoint signals are internal, not client SSE events, and do not end the
response. The host returns application output, not a graph-state snapshot or
public checkpoint references. Interrupt IDs identify pauses, not checkpoints.

### HTTP Errors and Response Failures

With branching enabled, field validation can fail before response execution.
For example, an empty `previous_response_id`, both non-null linkage fields, or
a body `response_id` returns HTTP 400 with an `invalid_request_error` envelope,
not a Responses response object.

An ineligible parent or unavailable required checkpoint can instead fail after
response execution has been admitted. A non-streaming request can return HTTP
200 with `status="failed"`. For example, if `resp_B` is still `in_progress`,
this request:

```json
{
  "model": "test", "previous_response_id": "resp_B",
  "input": "C", "store": true
}
```

returns an HTTP 200 response like:

```json
{
  "id": "resp_C",
  "previous_response_id": "resp_B",
  "status": "failed",
  "error": {
    "code": "server_error",
    "message": "The parent response must be stored and completed."
  }
}
```

With `stream=true`, the same failure can use HTTP 200 and end with an SSE
`response.failed` event instead of `response.completed`. The event's `response`
has `status="failed"` and includes the `error`.

Clients must handle HTTP errors and also check the returned response status or
terminal SSE event for accepted requests. HTTP 200 alone does not establish
success; inspect `response.error` when the response fails.

## Prerequisites and Limitations

Compile the graph with a history-preserving saver that implements asynchronous
checkpoint reads and writes, then opt in on the host:

```python
graph = builder.compile(checkpointer=saver)
server = ResponsesHostServer(graph, enable_response_branching=True)
```

The host reuses the graph-owned saver; it does not create one automatically.
`InMemorySaver` is suitable for local single-process use. Constructor validation
cannot certify a custom saver's history retention. The basic
[Responses hosting example][responses-example] shows server setup; checkpoint-based
branching additionally requires the saver and opt-in described in this guide.

The supported scope for checkpoint-based branching is single-worker execution:
one host worker process, not one active request at a time. Persistent stores
retain state across host recreation, but actual process-crash recovery and
multi-worker execution remain outside the supported guarantees.

Execution admission, duplicate-submission handling, task ownership, and recovery
scheduling belong to the upstream Agent Server SDK. The adapter does not enable
the SDK task manager itself. On the checked Core 2.2.0/Responses 2.3.0b2 runtime,
enable resilient tasks with `set_resilient_tasks_enabled(True)` from
`azure.ai.agentserver.core.tasks` before starting the host. Durable background
responses additionally require `ResponsesServerOptions(resilient_background=True)`
and persistent SDK task/response storage alongside graph checkpoints and branch
indexes. See the [Agent Server task framework][sdk-tasks]. Without an initialized
task manager, the checked SDK can fall back to non-durable in-process execution.

Admission behavior depends on SDK version and configuration. The adapter adds
no permanent response-ID deduplication, locks, leases, ownership tombstones, or
claim rollback. Reusing a completed response identity is not an exactly-once
retry contract; independent branches use new response identities.

### Parent Selection and Storage

- Parents must be created through the response-branching path, authorized,
  retained, and completed, with a readable exact root checkpoint and matching
  boundary and index records. Responses created with branching disabled or
  through explicit conversations are rejected as branch parents, even if they
  retain a checkpoint reference. Queued, in-progress, incomplete, failed, and
  cancelled responses are not eligible parents.
- Missing, inconsistent, deleted, or unreadable required root state fails
  explicitly. The root branching path never substitutes the latest root
  checkpoint, empty state, or reconstructed transcript. Parent root checkpoints
  are not deleted or rewritten by creating a child. Private subgraph storage
  and its failure handling remain application-owned.
- The branch store contains approval payloads as well as checkpoint references.
  Apply compatible access isolation and retention to it. Custom stores must
  accommodate the selected parent's pending approval payloads, which can exceed
  the protocol's minimum record capacity, or reject oversized writes explicitly
  without truncation.
- With `background=false`, `store=false` may use an eligible stored response as
  its parent, but the returned response cannot be the parent of a later
  checkpoint branch. Graph checkpoints may still be saved; `store=false` does
  not disable or delete them.
- `background=true` with `store=false` is rejected by the checked Responses SDK
  versions. The host does not change these settings or provide temporary
  response storage.
- With `enable_response_branching=True`, `steerable_conversations=True` and an
  existing `app=` are rejected when the server is created.
- Clients select a parent with `previous_response_id`. There are no new public
  `checkpoint_id`, `fork`, or `retry` fields. With branching enabled, a body
  `response_id` is rejected; existing Foundry identity headers remain supported.
  With branching disabled, request validation remains SDK-managed.

### Recovery and Approvals

- Task recovery uses the branching mode recorded when the task was accepted,
  even if `enable_response_branching` has changed since then. SDK scheduling
  decides whether recovery invokes the handler; the checked SDK can mark a
  stored foreground task failed rather than automatically resume its graph.
- With branching enabled, requests supplying neither `previous_response_id`
  nor `conversation` start independent root executions. A root need not be the
  user's first request. If it fails or the process crashes, the host does not
  automatically resume or rerun it. A client retry submits a new request with
  a new response identity and starts from scratch. Legacy root recovery is
  unchanged; actual process-crash recovery remains outside the supported
  guarantees.
- A normal HITL interrupt is not a root failure: the response can complete
  while the graph waits for matching resume or approval input.
- Recovery continues the same accepted task, not a new client fork request. Its
  saved starting point and progress determine where it resumes; see
  [Internal Design](#internal-design).
- Ordinary HITL approval, rejection, waiting, and partial approval remain
  supported. A response can be completed while its graph is paused. On the
  branching path, distinct responses selecting a paused parent have isolated
  root checkpoint pending writes and must each supply matching resume or approval
  input. One branch's answer does not globally consume the parent's pause or
  approve another branch. Local approval checks cover simple, sequential, and
  parallel interrupt graphs. Forking a paused subgraph is explicitly rejected:
  its separate checkpoint history and nested interrupt IDs are not transferred.
  Completed subgraph tasks in the same root step do not block branching at a
  root interrupt; their recorded outputs are reused without rerunning the tasks.
  This restriction does not change legacy continuation or completed-root forks.
  Completed-root forks still do not select or isolate private subgraph state.
- Completed functional `@task` calls inside a root node are also reused when
  that node pauses for approval, including repeated, nested, asynchronous, and
  parallel calls. Calls first reached after an approval still execute normally.
  Failed calls retain LangGraph's retry behavior; only successfully completed
  work is protected from this unintended replay.
- An `interrupt()` inside a functional `@task`, including a nested or
  asynchronous call, can be approved through either JSON or SSE responses.
  Waiting, sequential approvals, and partial approvals preserve the same
  branch-isolation and completed-task reuse behavior. Functional calls whose
  writes belong to the root checkpoint are not paused subgraphs: the restriction
  above still applies when a separate subgraph checkpoint history is involved.
- Checkpoints do not roll back or isolate external tools, shared stores, or
  files. Recovery may repeat unconfirmed work; applications remain responsible
  for replay-safe side effects. Exactly-once execution is not guaranteed.

### Instructions and Message State

Response branching leaves existing message conversion and instruction handling
unchanged. Top-level `instructions` are converted to a `SystemMessage`. With a
checkpointer, that message can persist in graph state and be restored with the
selected checkpoint, just like explicit system/developer input. Omitting
instructions, sending `null` or an empty string, or supplying new instructions
does not automatically remove an earlier instruction message. This existing
behavior does not provide the strict request-local instruction lifetime described
by the Responses API; correcting it is outside this branching feature's scope.

The host adds no instruction-source metadata, deletion tracking, or context-mode
transport. It does not reject an existing checkpoint merely because instruction
provenance is absent. Branch parents still require the response-boundary records
described above. Message reducers, summarization, trimming, and persistent
application prompts remain application-managed; branching restores their saved
state without imposing new message-identity requirements.

## Internal Design

Root graph checkpoints, response envelopes, and branch indexes are stored separately.
SDK task state and lifecycle remain SDK-managed. Durable deployments need stores with
compatible retention and user/deployment isolation. The table below covers
branch-execution failures and recovery of the same task, not a new client fork.
Terms refer to the current task:

- **Origin**: the confirmed starting checkpoint for the current task. For a
  paused parent, this is a new checkpoint in the parent's thread, with transferred
  pending writes and a persisted parent-to-fork interrupt ID mapping. Its resume
  writes do not modify the parent's checkpoint.
- **Recorded progress**: the checkpoint reference saved for the current task,
  not its parent's final checkpoint.
- **Published boundary**: the current response's own final checkpoint record,
  not the parent's. A completed response may have a boundary at a graph pause.
- **Branch index**: the separately stored copy of that boundary used to check
  that the response metadata and checkpoint mapping agree.

Paused-fork creation uses the strict saver for checkpoint reads and writes.
The config hook first selects the exact parent checkpoint; the host creates the
fork only after the complete `build_runnable_config()` override returns.
The host merges graph-bound defaults with that final request config, with
request values taking precedence. A request-local graph copy uses the effective
config without reapplying its original binding, preventing duplicate callbacks
and keeping execution consistent across supported LangGraph versions. The
shared application graph is not modified. Application-specific fields remain
available for fork reads, writes, functional-task identity records, and recovery
validation even when the saver returns only checkpoint reference fields.
Checkpoint reads also retain these configurable fields on ancestor references,
so channel-history reconstruction, including `DeltaChannel`, uses the same
application configuration without changing checkpoint identities. Pending-write
delegation preserves LangGraph's support for three-argument saver methods;
`task_path` is forwarded only when the corresponding method accepts it.
The copy shares the selected checkpoint's parent rather than extending its
history, so ancestor-reconstructed channels such as `DeltaChannel` do not replay
the transferred pending writes twice.
Root task and pending-write remapping is a pure transformation keyed by
`(task.name, task.path)`, keeping same-name fan-out tasks distinct.
Transferred writes retain their native task paths so channel-history reloads
preserve the same ordering as live execution, regardless of approval order.

Functional calls additionally need their call identity: a completed `@task`
result in pending writes does not contain its function name or full calling
context. Request-scoped runnable bindings capture the name, parent namespace,
call path, and step when LangGraph schedules a call, including cache hits that
skip execution callbacks, without storing task inputs.
The binding observes each copied node's combined runner while preserving its
original bound callable and writers. Native node-cache identities and entries
therefore remain compatible with unwrapped graphs, including shared caches.
Reused successful results are persisted before returning the task, including
on LangGraph versions that
only emit cached values. Before saving that task's writes,
the saver persists this identity in a reserved
`__langgraph_response_tasks__:<checkpoint_id>:<task_id>` namespace in the same
thread and backend. These are internal identity checkpoints, not application
subgraph state; applications must not use this namespace prefix. Retain them
for as long as their associated root checkpoint can be branched and prune them
alongside that checkpoint. History tools that list all namespaces also include
these records; select `checkpoint_ns=""` for root-only history. Normal
whole-thread deletion removes them with the other checkpoints.

When copying pending writes, the host uses LangGraph's internal task
construction helper rather than duplicating its ID hash algorithm. It first
reconstructs and verifies each original task ID, then generates the fork's IDs
in parent-before-child order before remapping pending writes. Interrupt IDs are
derived from the complete remapped task namespace, not only the root node name
and ID. An interrupt propagated to an ancestor retains its child's identity in
every write, and namespace ancestry is validated before copying it. Task-local
resume lists and ID-keyed global resume bindings are transferred consistently.
The host copies both identities and results. This also
preserves distinct calls to the same function with identical arguments. The
integration is compatibility-tested against the supported minimum LangGraph
version; missing helpers, unavailable or invalid identity records, and storage
errors fail explicitly rather than silently rerunning completed work.

Paused responses created by older hosts may lack these identity records. Start
new roots under the updated host or drain old workflows through an
application-approved continuation path; the host cannot safely infer the
missing identities from task IDs alone. This restriction
applies to paused checkpoints containing functional task writes, not ordinary
completed-parent branching or legacy continuation. Identity persistence adds
storage operations for functional calls but does not add an execution engine,
input-based result cache, or exactly-once guarantee.

Actual pause resumes and recovery from recorded progress use a request-scoped
checkpoint head. The execution config omits the explicit `checkpoint_id` that
some supported LangGraph versions interpret as replay, but the saver resolves
head reads to the exact selected checkpoint and advances only on this request's
root checkpoint writes. It never substitutes the thread's global latest
checkpoint. Explicit state reads and ordinary historical branch execution keep
their existing checkpoint-selection semantics.

Root checkpoint writes also record the current response's user-scoped identity
and confirmed origin in checkpoint metadata. A graph checkpoint event can arrive
before its write completes, including with synchronous durability. The host
waits for that exact root checkpoint to be read or successfully written before
publishing its reference as recovery progress. Write failures propagate without
publishing an unavailable checkpoint. Before resuming recorded progress,
the host verifies that this execution identity matches both the recovering
response and its origin. Sharing a thread or descending from the same parent is
not sufficient: sibling responses can share both. A paused fork is marked when
it is created, so valid progress may equal that response's paused origin without
losing completed pending writes. An ordinary completed parent's checkpoint is
not progress for a new child response. Private subgraph metadata is unchanged.

If B starts from its confirmed origin X and records progress at Y without publishing
its own boundary, recovery continues B from Y without adding B's input again.
With no progress or published boundary, B restarts from X with its original
input; unsaved work may run again. Recovery retains B's admitted mode and does
not fork its parent again. Same-thread fork checkpoints isolate LangGraph state;
they are not a separate execution-admission or task-management mechanism.
New origins use version 2 and require a valid resume mapping. Previously confirmed
version-1 origins remain recoverable without changing their recorded thread when
they have no recorded progress, or their progress has verifiable execution
ownership.

Older progress checkpoints without execution ownership cannot reliably be
distinguished from another response's progress. Recovery now fails explicitly
for such checkpoints instead of accepting them or replaying input. Before
upgrading a durable deployment, drain recoverable tasks whose progress was
recorded by an older host. Completed responses remain usable as branch parents;
they do not need to be relabelled. This validation adds no exactly-once or
execution-admission guarantee.

Paused origins also snapshot the parent's approval items. Before graph execution,
the host persists which snapshots remain unanswered, so recovery cannot reuse an
answered approval's identity for a later interrupt. Only unchanged pending
payloads inherit those identities; the published boundary carries their updated
wire-to-internal mapping. Older origins without snapshots remain recoverable but
do not gain the new identity guarantee retroactively.

Branching resumes address interrupts by ID even for a single reply, including
dictionary payloads. This avoids leaving a global resume write that could answer
a later interrupt again during recovery. Legacy resume conversion is unchanged.

Publishing an index or sending completion does not confirm terminal persistence.
Only a stored completed response with a matching index is reusable as a new-mode
parent.

| Persistence boundary or recovery state | Behavior on the branching path |
| --- | --- |
| Origin write fails or confirmed recovery origin is unavailable | Fail before graph execution. |
| Paused fork checkpoint or pending-write transfer fails | Fail before graph execution or origin confirmation; the parent remains unchanged. |
| Functional task identity is missing or invalid, or its read/write fails | Fail explicitly; do not rerun completed tasks or publish the failed branch as a parent. |
| Required graph checkpoint or state read/write fails | Fail explicitly; no latest-state, empty-state, or transcript fallback. |
| Approval snapshots or their ID bindings are invalid, or the unanswered-snapshot write fails | Fail before graph execution; do not silently replace the approvals or discard the storage error. |
| Confirmed origin; no recorded progress or published boundary | Restart the current task from its confirmed starting checkpoint with its original input; unsaved work may run again. |
| Valid recorded progress; no published boundary | Continue the current task from its saved progress without adding its input again. |
| Progress belongs to another response or origin, or its execution ownership is missing or invalid | Fail before graph execution or boundary publication; do not replay input or fall back to another checkpoint. |
| Boundary-index write fails | Fail the response; graph work may already have executed. |
| Boundary published; terminal persistence fails or the same task re-enters | Do not admit it as a parent without a stored completed response. Recovery fails before graph execution; no automatic terminal repair or fabricated completed response. |

The adapter validates opt-in linkage, records response/checkpoint mappings,
converts input and resume commands, executes the graph, and converts output. It
keeps no execution or pause ownership store and does not roll back SDK admission.
SDK leases, concurrency conflicts, recovery, and task cleanup follow the
configured SDK. Response/checkpoint retention remains a separate caller
responsibility.

## Validation Scope

The [branching tests][branching-tests] and [legacy hosting tests][hosting-tests]
use local graphs and stores; local HTTP tests run the real SDK host, not Foundry.

Coverage includes JSON/SSE `Send` fan-out with repeated node names, stable
unanswered approval items, independent fork answers, immutable response history,
and recovery with repeated prompts and string or dictionary resume values.
Functional-task branching was also exercised with LangGraph 1.1.1, including
reuse of completed task results after clearing the cache.

| Scenario | Expected behavior | Validation status |
| --- | --- | --- |
| Root creation | No `previous_response_id` or `conversation` starts an independent root. | Local HTTP tests passed. |
| Continuation and non-message state | Selecting B continues from B's exact root state, including its ledger, saved subgraph outputs, and application state references. | Local HTTP JSON/SSE tests passed. |
| Sibling branches | Selecting completed A while B is running or after B completes restores A's root state, not B's or the latest root state; B does not impose a sibling queue. | Local HTTP JSON/SSE tests passed for completed siblings; event-controlled local JSON/SSE probes passed with B still running, with SDK tasks disabled and enabled. |
| Application-owned subgraph state | Root state restores exactly while private persistent child memory can remain shared. Applications can instead load immutable child snapshots using references restored in root state. | Local JSON/SSE tests cover shared child memory, branch-local snapshots, concurrent siblings, immutable parent state, and explicit failure for a missing child snapshot. |
| Regeneration | Resubmitting B's input with A selected creates a new response ID from A's state. | Local HTTP JSON/SSE tests passed; identical model output is not guaranteed. |
| Conversations, branching disabled, or no checkpointer | Use legacy pointer/latest-state continuation, or message history without a saver; no exact-parent guarantee. | Local configuration, history, state-store, and HTTP regression tests passed. |
| JSON and SSE output | Return a new response, preserve its selected parent ID, and keep checkpoint signals private. | Local HTTP tests and documentation output probes passed. |
| Background execution | Reject an in-progress background parent; allow continuation and sibling branches after its stored completion. | Local event-controlled background HTTP test passed. |
| Interrupts and approvals | Complete the response while the graph pauses; matching input starts a new response in the same thread. Waiting, rejection, partial approvals, and independent historical answers preserve the parent pause. | Covered by local JSON/SSE, parallel waiting, MCP approval/rejection, and concurrent independent-approval tests; paused subgraph forks are rejected. |
| Stable approval identities | Unchanged unanswered approvals retain their IDs and original items; answered or changed approvals get new identities. Fork answers remain independent. | Local JSON/SSE, same-name `Send`, SQLite/File-store recreation, history, and simulated recovery tests passed, including storage-partition and item-default exceptions. |
| SDK execution admission | Duplicate submissions and task ownership follow the configured SDK, not adapter claims. | Adapter pass-through tests and local foreground JSON SDK probes passed. With tasks/resilient background off, overlapping duplicates execute; with both on, overlap returns 409. Completed-ID JSON requests can execute again in both configurations. |
| Response storage | Foreground `store=false` can consume a parent but cannot publish a reusable response; background with `store=false` is rejected. | Local HTTP storage and SDK validation tests passed. |
| Invalid requests | With branching enabled, reject invalid linkage, both non-null linkage fields, a body `response_id`, and unsupported host settings. With branching disabled, preserve existing SDK validation. | Local admission, disabled-path pass-through, and constructor tests passed. |
| Missing or ineligible parents/checkpoints | Reject legacy parents and fail explicitly without falling back to latest state, empty state, or message history. | Local JSON/SSE legacy-parent rejection tests and tests with missing/deleted records and backend failures passed. |
| Instructions and message compatibility | Preserve existing checkpointed instruction messages and application-managed reducers, summarization, and trimming. Do not enforce new instruction provenance or claim request-local isolation. | Local HTTP compatibility, formatted-message, old-checkpoint continuation, and real summarization/trim middleware tests passed. |
| Task recovery | Use the admitted mode and confirmed origin/progress; reuse the same-thread paused origin and its resume mapping without creating another fork; reject invalid state and a task that already published its boundary. | Local simulated recovery and invalid-mapping tests; no actual process termination/restart test. |
| Origin/index/terminal persistence failures | Fail safely and do not expose an unconfirmed or failed response as a reusable parent. | Local HTTP JSON/SSE fault injection passed; graph work may already have executed. |
| Host, saver, and SDK-store recreation | Retain exact response mappings and state across recreated objects. | Local SQLite saver and SDK local-store recreation passed; not a process-crash or failover test. |
| Live services, actual crashes, multi-worker execution, and operational lifecycle | Preserve graph isolation, SDK-managed execution/recovery, and compatible retention across persistent deployments. | Unverified; local failures or object recreation do not establish these guarantees. |

These results do not cover the full supported Python/dependency combinations.
The full minimum-dependency combination remains unverified.
SDK admission probes used Agent Server Core 2.2.0 and Responses 2.3.0b2, not a
minimum-dependency combination.

The supported contract is the Responses selection behavior above. Local passing
tests do not establish production/distributed recovery guarantees or full
OpenAI Responses API conformance. Parent selection follows the
[Responses create contract][openai-create]; checkpoint branching is the narrower
host capability described here, with the existing instruction-lifetime limitation
noted above.

[responses-example]: ../../../../samples/hosting/langgraph-hosted-agents/responses/01_basic/README.md
[branching-tests]: ../../tests/unit_tests/agents/hosting/test_response_branching.py
[hosting-tests]: ../../tests/unit_tests/agents/hosting/test_responses_host.py
[openai-create]: https://developers.openai.com/api/reference/resources/responses/methods/create
[sdk-tasks]: https://learn.microsoft.com/python/api/overview/azure/ai-agentserver-core-readme?view=azure-python#resilient-long-running-agents
