# Completed-Parent Response Branching

Checkpoint-based branching is an opt-in feature of `ResponsesHostServer`.
It is not available through `InvocationsHostServer`.

## Supported contract

Set `enable_response_branching=True` and compile the graph with a
history-preserving checkpointer that implements asynchronous reads and writes.
The host does not create or manage the graph's checkpointer.

| Request | Selected state | Behavior |
| --- | --- | --- |
| Branching enabled, no parent or explicit conversation | A new root thread | Execute the input and record its exact root checkpoint. |
| Branching enabled, `previous_response_id`, no explicit conversation | The exact checkpoint of a stored, completed, eligible parent | Continue, fork, or regenerate without selecting a sibling's latest state. |
| Branching enabled, explicit `conversation` | Existing conversation checkpoint selection | Preserve the existing conversation and approval behavior. |
| Branching disabled | Existing response-chain or conversation checkpoint selection | Preserve the existing default behavior, including partial approvals. |

An eligible parent must have been created with branching enabled and must have
matching stored response and checkpoint records. Its **graph execution must
also be finished**: no pending tasks or interrupts, no outstanding approvals,
and no function calls awaiting results. `response.status == "completed"` alone
is insufficient because a graph interrupted for approval can also produce a
completed response.

This initial scope deliberately excludes:

- Branching or resuming from a paused parent, even when the request contains
  all of its approval responses or tool results.
- Processing `mcp_approval_response` or `function_call_output` inputs on the
  response-branching path.
- Copying paused checkpoints, preserving approval IDs across partial replies,
  or remapping functional-task identities.
- Deciding whether partial approvals/results should be accumulated or rejected,
  or when such inputs should trigger further graph execution.

The existing default and explicit-conversation paths are unchanged. The new
branching path does not silently fall back to those paths for an unsupported
parent. Approval and tool-result aggregation can be addressed separately.

## Example

For a local single-worker host, replace the graph and host construction in the
basic hosting example with:

```python
from langgraph.checkpoint.memory import InMemorySaver

from langchain_azure_ai.agents.hosting import ResponsesHostServer

graph = create_agent(model, tools=[], checkpointer=InMemorySaver())
ResponsesHostServer(graph, enable_response_branching=True).run(port=8088)
```

From a separate client process:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8088", api_key="local", max_retries=0)
a = client.responses.create(input="Suggest a destination.", store=True)
b = client.responses.create(
    input="Plan a busy itinerary.", previous_response_id=a.id, store=True
)
c = client.responses.create(
    input="Plan a relaxed itinerary.", previous_response_id=a.id, store=True
)
regenerated_b = client.responses.create(
    input="Plan a busy itinerary.",
    previous_response_id=b.previous_response_id,
    store=True,
)
```

Both B and C start from A's root state, including non-message graph state.
C may start while B is still running; it does not select B's intermediate
checkpoint. Regeneration creates a new response and may produce different
output. Client-side undo/rewind selects an earlier response as the next parent;
it does not delete history, replay stored SSE events, or undo external effects.

## Failures and limitations

- Invalid linkage fields can return HTTP 400 before execution. A parent that is
  paused, unavailable, inconsistent, or not stored and completed produces a
  response with `status="failed"` or an SSE `response.failed` event. Check the
  response status or terminal event, not only the HTTP status.
- For example, if A returned a pending `mcp_approval_request`, a create request
  with `previous_response_id=A.id` fails on the branching path. Supplying a
  partial or complete approval batch does not make A eligible in this scope.
- Missing or mismatched exact checkpoints fail explicitly. There is no fallback
  to a thread's latest checkpoint.
- New branches select only the root namespace. Root state can contain subgraph
  outputs or application-owned snapshot references, but private persistent
  subgraph histories are not rewound or isolated by the host.
- Steering and an injected `ResponsesAgentServerHost` are unsupported.
- Checkpointed instruction messages retain their existing persistence behavior.
- `store=False` responses are not reusable parents. Background task and response
  storage policy remain SDK-owned.
- Supported deployment scope is a single worker. `InMemorySaver` loses state at
  shutdown; use a history-preserving durable saver for process recreation.
- Recovery uses the admitted mode, confirmed origin, and response-owned progress.
  An interrupted new root is not replayed automatically. A response with a
  published terminal boundary is not repaired or re-executed automatically.
- Exactly-once external effects, distributed execution, and actual
  process-crash windows are not guaranteed by this feature.

## Validation status

Local unit tests passed with frozen dependencies:

- Python 3.14.6: all 563 hosting tests.
- Python 3.11.16: all 133 branching and existing Responses-host tests.
- The 88 branching cases cover JSON/SSE, exact parent state and DeltaChannel
  history, running siblings, rejection of both complete and partial resume
  batches, unchanged legacy partial approvals, checkpoint failures, terminal
  storage failure, durable recreation, and response-owned recovery progress.
- Scoped Ruff lint/format and mypy checks passed.

Live Foundry execution, minimum dependency combinations, cross-worker execution,
and actual process-crash windows remain unverified. These tests do not establish
approval/result aggregation semantics or full Responses API conformance.
