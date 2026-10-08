# Checkpoint-Based Time Travel in Hosting

## Protocol Scope

`langchain-azure-ai` supports both `ResponsesHostServer` and
`InvocationsHostServer` as hosting protocols. Checkpoint-based time travel over
HTTP is available only through Responses with `enable_response_branching=True`.
The option defaults to `False`.

Invocations supports checkpointed session continuation and configured task
recovery, not user-selected historical checkpoints. Its `previous_invocation_id`
is an ordering precondition against the latest accepted turn, not a checkpoint
selector. Native LangGraph APIs on `host.graph` are separate from this HTTP
contract.

## Responses Behavior

For checkpointed graphs, fresh requests follow the paths below. A conversation
or legacy chain pointer is mutable; it is not an immutable response boundary.

| Configuration and linkage                                     | Starting checkpoint/state                                                                                                   | Behavior                                                                                                                            |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Branching enabled, `previous_response_id`, no `conversation`  | The exact checkpoint recorded for the authorized, stored, completed parent response                                         | Run the new input from that boundary; support continuation, sibling forks, and regeneration without changing the parent checkpoint. |
| Branching enabled, no parent or conversation                  | A new response-scoped thread with no prior checkpoint                                                                       | Start an independent root. Omitting the parent or setting it to `null` has the same effect.                                         |
| Either branching setting, explicit `conversation`, no parent  | The conversation's saved checkpoint pointer, or the resolved conversation thread's latest checkpoint when no pointer exists | Use legacy conversation continuation, not response-boundary time travel. A new conversation starts without saved state.             |
| Branching disabled, `previous_response_id`, no `conversation` | The legacy chain's saved checkpoint pointer, or the ancestry-resolved thread's latest checkpoint when no pointer exists     | Continue legacy graph state; the parent ID does not guarantee restoration of that response's historical checkpoint.                 |
| Branching disabled, no parent or conversation                 | A new response-scoped thread with no prior checkpoint                                                                       | Start a new root using the legacy path.                                                                                             |
| Either setting, both linkage fields non-null                  | None                                                                                                                        | Reject before execution; neither linkage field takes precedence.                                                                    |

Without a graph checkpointer, the legacy Responses path reconstructs message
history instead of graph state. Transcript branching is not checkpoint-based
time travel. Enabling branching without a usable saver fails at construction.
Recovery of an admitted task follows its recorded mode, not a later flag change.

## Meaning of Time Travel

Time travel here means fork-like continuation from a completed parent response's
checkpoint, including non-message graph state, with new request input. If B was
created from A, another request selecting A starts from A, not from B or the
thread's latest state. Selecting B continues after B; it does not rerun B.

Regenerating B means submitting its input again with parent A and receiving a
new response ID. This does not promise identical model output, idempotent HTTP
retry, or exactly-once tool effects. Requests send only additional input, not a
duplicate transcript of the selected parent's history.

This contract does not expose arbitrary checkpoint selection, parent-execution
replay, or arbitrary state editing. SSE event replay and recovery of the same
interrupted task are separate operations, not new historical branches.

## Prerequisites and Limitations

Compile the graph with a history-preserving saver that implements asynchronous
checkpoint reads and writes, then opt in on the host:

```python
graph = builder.compile(checkpointer=saver)
server = ResponsesHostServer(graph, enable_response_branching=True)
```

The host reuses the graph-owned saver; it does not create one automatically.
`InMemorySaver` is suitable for local single-process use. Durable deployments
need persistent graph, response, task, and reference/ownership stores with
compatible retention and user/deployment isolation. Constructor validation
cannot certify a custom saver's history retention. See the
[Responses example][responses-example] for setup.

### Parent Selection and Storage

- Parents must be authorized, retained, and completed, with a readable exact graph
  checkpoint. New-mode parents require matching boundary and index records.
  Legacy parents may instead use a verified per-response checkpoint reference
  from stored response metadata without a new boundary/index. Queued, in-progress,
  incomplete, failed, and cancelled responses are not eligible parents.
- Missing, inconsistent, deleted, or unreadable required state fails explicitly.
  The branching path never substitutes the latest checkpoint, empty state, or
  reconstructed transcript. Parent checkpoints are not deleted or rewritten by
  creating a child.
- Foreground `store=false` can consume an eligible parent but does not publish
  its own reusable response boundary. It does not erase saver history. The
  checked Responses SDK versions reject `background=true, store=false`; the
  host does not silently enable storage or implement temporary retention for
  that combination.
- `steerable_conversations=True` and attaching an existing `app` are unsupported
  when branching is enabled and are rejected explicitly. No new public
  `checkpoint_id`, `fork`, `retry`, or body `response_id` field is provided;
  Foundry platform identity headers remain supported.

### Recovery and Approvals

- Failed or crashed roots admitted on the branching path are not automatically
  resumed or replayed. A client retry creates a new root. Legacy root recovery
  is unchanged.
- Parent-linked task recovery uses its own confirmed execution checkpoint, or
  its confirmed parent origin if no execution progress was durably recorded.
  Required recorded state becoming unavailable is a failure, not permission to
  start again from a different checkpoint. Production crash guarantees remain
  unverified as described below.
- Ordinary HITL approval, rejection, waiting, and partial approval remain
  supported. A response can be completed while its graph is paused. On the
  branching path, independent historical or competing approval forks are
  unsupported; a consumed pause cannot be resumed with a different answer.
- Checkpoints do not roll back or isolate external tools, shared stores, or
  files. Recovery may repeat unconfirmed work; applications remain responsible
  for replay-safe side effects. Exactly-once execution is not guaranteed.
- Branching ownership records do not expire automatically. Only confirmed
  pre-admission rejection can release a claim, with owner and ETag checks.
  Accepted or uncertain requests retain ownership; no general cleanup or
  compaction is supplied. Removing consumed ownership records can permit
  duplicate execution.

<!-- markdownlint-disable-next-line MD033 -->

<a id="request-instruction-isolation-2026-09-30"></a>

### Request Instructions

Top-level `instructions` apply only to the current response; omission, `null`,
or an empty string does not inherit the parent's value. Explicit system/developer
input and application prompts are preserved. Recovery retains the admitted
request's instructions. These rules and linkage validation also apply when
branching is disabled; callers must resend persistent top-level instructions and
choose only one non-null linkage field.

The default `instructions_mode="messages"` supports verified explicit removal by
summarization/trimming, but instructions can enter summarizer input or summaries.
Opt-in `instructions_mode="context"` keeps raw instructions out of graph state
and summarizer input; applications must integrate `ResponsesInstructionsMiddleware`
or `get_response_instructions(config)` in their model calls. Keep that integration
on every worker that may recover context-mode tasks. Unverifiable instruction
provenance fails closed rather than rewriting old checkpoints. See the
[instruction example][instructions-example].

## Validation Status

| Local check                                                              | Result                                                                                                                                                                                       |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Full hosting suite, Python 3.14.6 with frozen dependencies               | 719 passed, including 231 branching cases.                                                                                                                                                   |
| Branching suite, Python 3.11.16 with minimum direct hosting dependencies | 231 passed; LangChain 1.2.12, LangGraph 1.1.1, prebuilt 1.0.8, Agent Server Core/Responses 2.1.0b2, and Invocations 1.1.0b1. Transitive dependencies were not all at their minimum versions. |
| Ruff, formatting, and mypy for tests and changed runtime modules         | Passed.                                                                                                                                                                                      |

Local tests cover root creation, continuation, forks, regeneration, non-message
state, concurrent isolation, JSON/SSE/background execution, exact-read failures,
instruction lifetime, ordinary HITL, and admission/rollback races. SQLite saver
and SDK local-store recreation passed; replacing host/store objects is not a
process-crash or distributed-failover test. These are recorded validation
results, not a full Python/dependency matrix or proof of production readiness.

Production and distributed recovery remain unverified, including:

- Live Foundry storage and persistent savers, cross-worker branches and
  competing approvals, and deployment/user isolation and retention behavior.
- Actual process termination/restart, first durable SDK admission, and failures
  between graph, origin/index, and terminal response persistence. Local checks
  do not establish repair without rerunning graph work across these boundaries.
- Ownership-record maintenance, long-running cancellation/reconnection, and
  the full supported Python/dependency combinations.

The supported contract is the Responses selection behavior above. Local passing
tests do not establish production/distributed recovery guarantees or full
OpenAI Responses API conformance. Public fields follow the
[Responses create contract][openai-create]; checkpoint branching is the narrower
host capability described here.

[responses-example]: ../../../../samples/hosting/langgraph-hosted-agents/responses/01_basic/README.md#opt-in-checkpoint-branches
[instructions-example]: ../../../../samples/hosting/langgraph-hosted-agents/responses/01_basic/README.md#request-instructions-with-summarization
[openai-create]: https://developers.openai.com/api/reference/resources/responses/methods/create
