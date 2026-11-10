# Response branching

Response branching lets a new response start from an earlier response's saved
LangGraph state. Set `enable_response_branching=True` on `ResponsesHostServer`
and select the parent with `previous_response_id`.

This document describes the branching design, including the
[open implementation and compatibility work](#open-items). See
[Responses hosting](responses.md) for the shared protocol and host configuration.

## 1. Behavior

### Branch state

Each child starts from its parent's exact graph checkpoint and adds its own
input. For example, B and C can explore two plans from the same answer A:

```text
A: Suggest a destination.
|-- B: Plan a busy itinerary.
`-- C: Plan a relaxed itinerary.
```

Both requests use `previous_response_id=A.id`. B sees A+B, and C sees A+C,
including when B finishes before C starts. The parent stays unchanged, and each
child can become a parent for later responses. Siblings can run concurrently
within one host process.

Branches share a LangGraph thread with separate checkpoint histories. The host
restores prior state from the selected checkpoint and passes only the current
request input to the graph.

### Requirements

| Requirement | Configuration or condition |
| --- | --- |
| Branching enabled | `enable_response_branching=True`; the default is `False`. |
| Graph checkpointer | A history-preserving saver with asynchronous reads and writes. |
| Stored parent | A response available to the caller, with `status="completed"` and an exact checkpoint reference. |
| Finished execution | The graph has no pending tasks or interrupts, and the response has no unanswered function calls or MCP approval requests. |
| Host configuration | `app=None` and `options.steerable_conversations=False`. |

Unsupported host configurations raise `ValueError` during initialization. Use
an async saver such as `AsyncSqliteSaver` for SQLite, and a durable saver when
state must survive a restart.

> [!IMPORTANT]
> Branching from a paused graph is currently unsupported, even when the response
> status is `completed`. To submit tool results or approval decisions, use
> [ordinary continuation](responses.md#human-in-the-loop) with branching disabled.

Eligible parents can come from ordinary continuation or an existing
conversation, including responses created before branching was enabled.

### Request linkage

| Request fields | Behavior |
| --- | --- |
| Neither `previous_response_id` nor `conversation` | Start a root response in a new thread. |
| `previous_response_id` only | Start a child from that response's exact checkpoint. |
| `conversation` only | Continue the conversation through the ordinary continuation path. |
| Both fields | Return HTTP 400. |

Omitted or `null` linkage fields leave that linkage unset. Parent lookup uses
the caller's authorized storage scope. A missing or invalid parent causes an
error before graph execution.

Public request semantics follow the [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)
and [conversation-state guide](https://developers.openai.com/api/docs/guides/conversation-state).
Top-level `instructions` apply to the current request, so clients resend them
on each request that needs them. System and developer message items remain part
of graph history. Tool outputs and approval decisions use the ordinary
continuation rules.

### Errors and limits

Admission errors return JSON before streaming or graph execution:

| Condition | Result |
| --- | --- |
| A checkpointed graph receives `store=False` | HTTP 400, `invalid_request_error`, `param="store"`. Omit `store` or use `store=True`. |
| Parent is missing, inaccessible, or has no valid thread in the caller's scope | HTTP 404, `previous_response_not_found`. |
| Parent is unfinished, or its checkpoint reference or checkpoint is invalid, missing, or mismatched | HTTP 400, `invalid_request_error`, `param="previous_response_id"`. |
| Unexpected provider or checkpointer failure during admission | HTTP 500, `server_error`, with internal details logged. |

The `store=False` rule applies to every request with a graph checkpointer,
including roots and ordinary continuation. Requests without a checkpointer
follow the [shared storage rules](responses.md#3-conversation-and-state).

After admission, execution or storage failures produce `status="failed"` or an
SSE `response.failed` event. Clients check the terminal response or event even
when the HTTP request succeeds. The selected checkpoint must remain available
throughout execution.

The supported deployment uses one host process and one worker. Branching
restores the root checkpoint namespace (`checkpoint_ns=""`); private persistent
subgraph histories retain their own lifecycle. External effects can repeat
during execution or recovery, so applications should make replayed work safe.

## 2. Internal data structures and flow

### Storage

The response provider and graph saver retain different parts of a response:

| Data | Storage | Configuration |
| --- | --- | --- |
| Response status, input/output, and internal checkpoint references | SDK response provider | Host default, or server `store=provider`. |
| Graph state, pending writes, and checkpoint history | LangGraph checkpoint saver | Graph `checkpointer=...`. |

The host defaults to `FileResponseStore` locally and `FoundryStorageProvider`
on Foundry. The constructor's `store` selects a provider; the request's `store`
is a retention boolean that defaults to `True`.

Keep both the response and the checkpoint history available for as long as the
response can be reused. The application owns the saver's lifecycle and retention.
Response deletion leaves graph checkpoints intact.

### Checkpoint references

A checkpoint reference identifies a thread and checkpoint in the root namespace.
`CheckpointRef` holds `thread_id` and `checkpoint_id`; `TaskStorageManager`
persists them as `langgraph_thread_id` and `langgraph_checkpoint_id` in internal
SDK metadata.

| Reference role | Meaning |
| --- | --- |
| Origin | The parent's final checkpoint, used to start a child. |
| Progress | The current response's latest saved graph checkpoint, used for recovery. |
| Final | The last progress reference retained with the completed response, used by later children. |

These roles point to graph state in the saver. Several responses can share a
thread, so recovery must select the current response's progress. The parent's
checkpoint supplies the child's starting state; the child records its own
progress as it runs. Thread IDs are scoped to the platform user when that
identity is available.

### Fresh request flow

1. **Select the parent.** Resolve `previous_response_id` through the response
   provider in the caller's storage scope. Reuse the lookup within the request.
2. **Validate the starting state.** Check the parent's status, output, and exact
   checkpoint, then verify that the restored graph has finished.
3. **Run the graph.** Supply the current input and instructions, using the
   selected checkpoint for prior state. Checkpointed execution uses
   `durability="sync"` and streams graph updates, messages, and checkpoints.
4. **Save progress.** After a graph checkpoint is saved, attach its reference
   to task metadata, flush output events, and request an SDK recovery snapshot
   with `stream.checkpoint()`.
5. **Finalize the response.** Emit the terminal response for SDK storage. A
   completed, stored response retains its final checkpoint reference for reuse.

Graph checkpoints and SDK response snapshots are saved separately. Recovery
uses the last snapshot the SDK persisted, which can lag behind graph execution.
A reusable parent needs both a stored response and its restorable checkpoint.

Branching uses the shared input conversion, graph streaming, progress storage,
and finalization path. The conversation-chain store supplies the checkpoint
for ordinary linear continuation; response-ID branches select their parent's
stored reference. The interaction between these paths is tracked below.

### Recovery flow

The Agent Server SDK schedules recovery for stored background work when
`resilient_background=True`. Response/task storage and graph checkpoints must
survive the restart. See [restart recovery](responses.md#restart-recovery) for
the shared setup.

| Persisted state | Recovery action |
| --- | --- |
| Root with no graph progress | Replay the original request in its existing thread. |
| Child with no graph progress | Reload and validate the parent, then replay the child's input. |
| Root or child with valid progress for that response | Resume the saved checkpoint with input `None`, preserving pending writes. |
| Saved graph already finished | Finalize the response using the saved graph state. |
| Progress is malformed, unavailable, mismatched, or belongs to another response | Fail recovery. |

Valid response progress provides the recovery starting point without another
parent lookup. The output stream starts from the persisted response snapshot.
Work after the last saved snapshot, including external effects and response
finalization, can run again.

### Source entry points

Implementation details live with the code:

- [Responses host](../../langchain_azure_ai/agents/hosting/_responses_host.py):
  configuration, execution, and recovery.
- [Request preparation](../../langchain_azure_ai/agents/hosting/_responses/request_preparation.py):
  parent lookup and admission checks.
- [Checkpoint references](../../langchain_azure_ai/agents/hosting/_responses/checkpoint_ref.py)
  and [task storage](../../langchain_azure_ai/agents/hosting/_responses/task_storage_manager.py):
  references attached to each response.
- [Graph-stream conversion](../../langchain_azure_ai/agents/hosting/_converters/_stream.py):
  output and progress snapshots.
- [Branching tests](../../tests/unit_tests/agents/hosting/test_response_branching.py):
  parent preservation, concurrent siblings, failures, and recovery.

### Open items

The remaining implementation and compatibility work is:

- Validate parent outputs for unanswered tool calls as well as checking graph
  completion, and reject requests combining both linkage fields with HTTP 400.
- Verify request-scoped `instructions` across checkpoint restoration.
- Cover admission errors through HTTP, including status, error code, parameter,
  and absence of graph execution. Confirm remaining reference-error details
  against the service.
- Verify default local and Foundry providers, explicit provider overrides,
  eligible parents created through conversations, and ordinary tool/approval
  continuation.
- Define conversation linkage: if A belongs to a conversation and B branches
  from A, decide which context a later request using that conversation selects.

Branching from paused parents requires a separate design for isolating tool
replies, approval decisions, and resume state between children.

## 3. Use cases

### Set up a local server

Install `langchain-azure-ai[hosting]` and configure `model` as in the
[basic hosting setup](../../README.md#hosting-langgraph-agents-on-microsoft-foundry).
Use an in-memory saver for this local example:

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

### Compare alternatives

Run the client in a separate process. Both plans start from the destination
suggested by A:

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

### Continue, regenerate, or rewind

| Goal | Request |
| --- | --- |
| Continue B | Send new input with `previous_response_id=b.id`. |
| Regenerate B | Resend B's input with `previous_response_id=a.id` to create another child of A. |
| Rewind | Select an earlier response as the parent and send new input from that point. |

Rewinding creates a new response while preserving existing history. External
effects from earlier execution remain in place.
