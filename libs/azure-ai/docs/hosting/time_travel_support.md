# Response Branching in Hosting

## Protocol Scope

`langchain-azure-ai` supports both `ResponsesHostServer` and
`InvocationsHostServer` as hosting protocols. Response branching over
HTTP is available only through Responses with `enable_response_branching=True`.
The option defaults to `False`.

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
| Branching enabled, `previous_response_id`, no `conversation` | The exact checkpoint recorded for the authorized, stored, completed parent response | Run the new input from that boundary; support continuation, sibling forks, and regeneration without changing the parent checkpoint. |
| Branching enabled, neither `previous_response_id` nor `conversation` supplied | A new response-scoped thread with no prior checkpoint | Start an independent root. Omitting a linkage field or setting it to `null` has the same effect. |
| Either branching setting, explicit `conversation`, no `previous_response_id` | The conversation's saved checkpoint pointer, or the resolved conversation thread's latest checkpoint when no pointer exists | Use legacy conversation continuation, not response-boundary branching. A new conversation starts without saved state. |
| Branching disabled, `previous_response_id`, no `conversation` | The legacy chain's saved checkpoint pointer, or the ancestry-resolved thread's latest checkpoint when no pointer exists | Continue legacy graph state; the parent ID does not guarantee restoration of that response's historical checkpoint. |
| Branching disabled, neither `previous_response_id` nor `conversation` supplied | A new response-scoped thread with no prior checkpoint | Start a new root using the legacy path. |
| Either setting, both linkage fields non-null | None | Reject before execution; neither linkage field takes precedence. |

Without a graph checkpointer, `enable_response_branching` must remain `False`.
Responses restores only message history, not other graph state such as counters
or application variables.

## Meaning of Response Branching

Response branching backed by LangGraph checkpoints follows the
[OpenAI Responses parent-selection model][openai-create]: `previous_response_id`
selects a response boundary, not an arbitrary or intermediate graph checkpoint.
The API does not expose checkpoint selection or state editing.

If B was created from A, another request selecting A starts from A, not B or the
thread's latest state. Selecting B continues after B; it does not rerun B.
Restored state includes non-message values. Send only new input, not a duplicate
transcript of the selected response's history.

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
has answered. Resume this custom interrupt with a new request selecting P:

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

Checkpoint signals are internal, not client SSE events, and do not end the
response. The host returns application output, not a graph-state snapshot or
public checkpoint references. Interrupt IDs identify pauses, not checkpoints.

## Prerequisites and Limitations

Compile the graph with a history-preserving saver that implements asynchronous
checkpoint reads and writes, then opt in on the host:

```python
graph = builder.compile(checkpointer=saver)
server = ResponsesHostServer(graph, enable_response_branching=True)
```

The host reuses the graph-owned saver; it does not create one automatically.
`InMemorySaver` is suitable for local single-process use. Constructor validation
cannot certify a custom saver's history retention. See the
[Responses example][responses-example] for setup.

The supported scope for checkpoint-based branching is single-worker execution.
Persistent stores retain state across host recreation, but actual process-crash
recovery and multi-worker execution remain outside the supported guarantees.

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
  retained, and completed, with a readable exact graph checkpoint and matching
  boundary and index records. Responses created with branching disabled or
  through explicit conversations are rejected as branch parents, even if they
  retain a checkpoint reference. Queued, in-progress, incomplete, failed, and
  cancelled responses are not eligible parents.
- Missing, inconsistent, deleted, or unreadable required state fails explicitly.
  The branching path never substitutes the latest checkpoint, empty state, or
  reconstructed transcript. Parent checkpoints are not deleted or rewritten by
  creating a child.
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
  `checkpoint_id`, `fork`, or `retry` fields. A body `response_id` is rejected;
  existing Foundry identity headers remain supported.

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
  checkpoint pending writes and must each supply matching resume or approval
  input. One branch's answer does not globally consume the parent's pause or
  approve another branch. Local approval checks cover simple, sequential, and
  parallel interrupt graphs; copying separate nested-subgraph checkpoint
  histories has not been validated.
- Checkpoints do not roll back or isolate external tools, shared stores, or
  files. Recovery may repeat unconfirmed work; applications remain responsible
  for replay-safe side effects. Exactly-once execution is not guaranteed.

<!-- markdownlint-disable-next-line MD033 -->
<a id="request-instruction-isolation-2026-09-30"></a>

### Request Instructions

Top-level `instructions` are supplied for the current response and are not
automatically reused by later requests. Omission, `null`, or an empty string
does not inherit the parent's value. In `messages` mode, instruction content
may remain in summaries after the original instruction message is removed.
Explicit system/developer input and application prompts are preserved. Recovery
retains the admitted request's instructions. These rules and linkage validation
also apply when branching is disabled; callers must resend persistent top-level
instructions and choose only one non-null linkage field.

Both instruction modes support branching. The default
`instructions_mode="messages"` supports verified explicit removal by
summarization/trimming, but instructions can enter summarizer input or summaries.
Opt-in `instructions_mode="context"` keeps raw instructions out of graph messages
and summarizer input; applications must integrate `ResponsesInstructionsMiddleware`
or `get_response_instructions(config)` in their model calls. It does not remove
the influence of instructions on earlier model output. Keep that integration on
every worker that may recover context-mode tasks. Unverifiable instruction
provenance fails closed rather than rewriting old checkpoints. See the
[instruction example][instructions-example].

## Internal Design

Graph checkpoints, response envelopes, and branch indexes are stored separately.
SDK task state and lifecycle remain SDK-managed. Durable deployments need stores with
compatible retention and user/deployment isolation. The table below covers
branch-execution failures and recovery of the same task, not a new client fork.
Terms refer to the current task:

- **Origin**: the confirmed starting checkpoint for the current task. For a
  paused parent, this is a response-scoped copy with the same checkpoint and
  interrupt IDs and its pending writes, not the parent's mutable resume writes.
- **Recorded progress**: the checkpoint reference saved for the current task,
  not its parent's final checkpoint.
- **Published boundary**: the current response's own final checkpoint record,
  not the parent's. A completed response may have a boundary at a graph pause.
- **Branch index**: the separately stored copy of that boundary used to check
  that the response metadata and checkpoint mapping agree.

If B starts from its confirmed origin X and records progress at Y without publishing
its own boundary, recovery continues B from Y without adding B's input again.
With no progress or published boundary, B restarts from X with its original
input; unsaved work may run again. Recovery retains B's admitted mode and does
not copy its parent again. Paused checkpoint copies isolate LangGraph state;
they are not a separate execution-admission or task-management mechanism.

Publishing an index or sending completion does not confirm terminal persistence.
Only a stored completed response with a matching index is reusable as a new-mode
parent.

| Persistence boundary or recovery state | Behavior on the branching path |
| --- | --- |
| Origin write fails or confirmed recovery origin is unavailable | Fail before graph execution. |
| Paused checkpoint copy or pending-write copy fails | Fail before graph execution or origin confirmation; the parent remains unchanged. |
| Required graph checkpoint or state read/write fails | Fail explicitly; no latest-state, empty-state, or transcript fallback. |
| Confirmed origin; no recorded progress or published boundary | Restart the current task from its confirmed starting checkpoint with its original input; unsaved work may run again. |
| Valid recorded progress; no published boundary | Continue the current task from its saved progress without adding its input again. |
| Boundary-index write fails | Fail the response; graph work may already have executed. |
| Boundary published; terminal persistence fails or the same task re-enters | Do not admit it as a parent without a stored completed response. Recovery fails before graph execution; no automatic terminal repair or fabricated completed response. |

The adapter validates linkage, records response/checkpoint mappings, converts
input and resume commands, executes the graph, and converts output. It keeps no
execution or pause ownership store and does not roll back SDK admission. SDK
leases, concurrency conflicts, recovery, and task cleanup follow the configured
SDK. Response/checkpoint retention remains a separate caller responsibility.

## Validation Status

The latest full hosting run uses Python 3.14.6, Agent Server Core 2.2.0,
Responses 2.3.0b2, and LangGraph 1.2.12. The minimum-dependency result below
is an earlier baseline; the latest follow-up regression cases have not been
rerun on that combination. Adapter-ownership tests were removed rather than
reimplemented; checkpoint-copy safeguards were added.

| Local check                                                              | Result                                                                                                                                                                                       |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Full hosting suite, Python 3.14.6 with frozen dependencies               | 726 passed, including 238 branching cases.                                                                                                                                                   |
| Branching suite, Python 3.11.16 with minimum direct hosting dependencies | 231 passed; LangChain 1.2.12, LangGraph 1.1.1, prebuilt 1.0.8, Agent Server Core/Responses 2.1.0b2, and Invocations 1.1.0b1. Transitive dependencies were not all at their minimum versions. |
| Ruff, formatting, and mypy for tests and changed runtime modules         | Passed.                                                                                                                                                                                      |

The [branching tests][branching-tests] and [legacy hosting tests][hosting-tests]
use local graphs and stores; local HTTP tests run the real SDK host, not Foundry.

| Scenario | Expected behavior | Validation status |
| --- | --- | --- |
| Root creation | No `previous_response_id` or `conversation` starts an independent root. | Local HTTP tests passed. |
| Continuation and non-message state | Selecting B continues from B's exact state, including its ledger or other graph values. | Local HTTP JSON/SSE tests passed. |
| Sibling branches | Selecting A after B completes restores A, not B or the latest thread state. | Local HTTP JSON/SSE tests passed. |
| Regeneration | Resubmitting B's input with A selected creates a new response ID from A's state. | Local HTTP JSON/SSE tests passed; identical model output is not guaranteed. |
| Conversations, branching disabled, or no checkpointer | Use legacy pointer/latest-state continuation, or message history without a saver; no exact-parent guarantee. | Local configuration, history, state-store, and HTTP regression tests passed. |
| JSON and SSE output | Return a new response, preserve its selected parent ID, and keep checkpoint signals private. | Local HTTP tests and documentation output probes passed. |
| Background execution | Reject an in-progress background parent; allow continuation and sibling branches after its stored completion. | Local event-controlled background HTTP test passed. |
| Interrupts and approvals | Complete the response while the graph pauses; matching input starts a new response. Waiting, rejection, partial approvals, and independent historical answers preserve the parent pause. | Local JSON/SSE and concurrent independent-approval tests passed for simple, sequential, and parallel interrupt graphs; nested-subgraph copying remains unverified. |
| SDK execution admission | Duplicate submissions and task ownership follow the configured SDK, not adapter claims. | Adapter pass-through tests and local foreground JSON SDK probes passed. With tasks/resilient background off, overlapping duplicates execute; with both on, overlap returns 409. Completed-ID JSON requests can execute again in both configurations. |
| Response storage | Foreground `store=false` can consume a parent but cannot publish a reusable response; background with `store=false` is rejected. | Local HTTP storage and SDK validation tests passed. |
| Invalid requests | Reject invalid linkage, both non-null linkage fields, a body `response_id`, and unsupported host settings. | Local admission and constructor tests passed. |
| Missing or ineligible parents/checkpoints | Reject legacy parents and fail explicitly without falling back to latest state, empty state, or message history. | Local JSON/SSE legacy-parent rejection tests and tests with missing/deleted records and backend failures passed. |
| Instructions | Do not automatically reuse top-level instructions. Preserve verified message removal and context-mode model-call integration without promising removal of summary/output influence. | Local HTTP tests and real summarization/trim middleware tests passed. |
| Task recovery | Use the admitted mode and confirmed origin/progress; reuse an isolated paused origin without recopying; reject invalid state and a task that already published its boundary. | Local simulated recovery passed; no actual process termination/restart test. |
| Origin/index/terminal persistence failures | Fail safely and do not expose an unconfirmed or failed response as a reusable parent. | Local HTTP JSON/SSE fault injection passed; graph work may already have executed. |
| Host, saver, and SDK-store recreation | Retain exact response mappings and state across recreated objects. | Local SQLite saver and SDK local-store recreation passed; not a process-crash or failover test. |
| Live services, actual crashes, multi-worker execution, and operational lifecycle | Preserve graph isolation, SDK-managed execution/recovery, and compatible retention across persistent deployments. | Unverified; local failures or object recreation do not establish these guarantees. |

These results do not cover the full supported Python/dependency combinations.
The older minimum-dependency baseline has not been rerun for the latest cases.
The validation table describes the current implementation. SDK admission probes
used the latest runtime above, not the earlier minimum-dependency combination.

The supported contract is the Responses selection behavior above. Local passing
tests do not establish production/distributed recovery guarantees or full
OpenAI Responses API conformance. Public fields follow the
[Responses create contract][openai-create]; checkpoint branching is the narrower
host capability described here.

[responses-example]: ../../../../samples/hosting/langgraph-hosted-agents/responses/01_basic/README.md#opt-in-checkpoint-branches
[instructions-example]: ../../../../samples/hosting/langgraph-hosted-agents/responses/01_basic/README.md#request-instructions-with-summarization
[branching-tests]: ../../tests/unit_tests/agents/hosting/test_response_branching.py
[hosting-tests]: ../../tests/unit_tests/agents/hosting/test_responses_host.py
[openai-create]: https://developers.openai.com/api/reference/resources/responses/methods/create
[sdk-tasks]: https://learn.microsoft.com/python/api/overview/azure/ai-agentserver-core-readme?view=azure-python#resilient-long-running-agents
