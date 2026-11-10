# Response branching

Response branching lets a new response start from an earlier response's saved
LangGraph state. Set `enable_response_branching=True` on `ResponsesHostServer`
and select the parent with `previous_response_id`.

This document describes the branching design, including the
[follow-up implementation and compatibility work](#follow-up). See
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

Branching adds exact parent selection and finished-parent checks to the shared
checkpointed execution path. It shares storage, graph execution, output
conversion, and recovery from saved progress with ordinary continuation.

### Shared storage

Both paths use the response provider and graph saver for the same data:

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
Both paths use `CheckpointRef` for `thread_id` and `checkpoint_id`, and
`TaskStorageManager` persists them as `langgraph_thread_id` and
`langgraph_checkpoint_id` in internal SDK metadata. Branching uses the parent's
final reference as the child's origin:

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

The current checkpointed paths differ during parent selection and validation:

| Stage | Ordinary checkpointed continuation | Branching |
| --- | --- | --- |
| Select starting state | Read the conversation-chain checkpoint when available; otherwise use the thread's latest checkpoint. | Load the named parent's exact final checkpoint from its stored response reference. |
| Validate graph state | A paused graph can accept matching tool results or approval decisions. | Require a completed parent response and a finished graph under the parent eligibility rules. |

Between selection and finalization, both paths run the graph with current input
and instructions, using the selected checkpoint for prior state. Execution uses
`durability="sync"` and streams graph updates, messages, and checkpoints. As graph
checkpoints are saved, the host attaches their references to task metadata,
flushes output events, and requests SDK recovery snapshots with
`stream.checkpoint()`. The SDK then stores the terminal response with its final
checkpoint reference.

Graph checkpoints and SDK response snapshots are saved separately. Recovery
uses the last snapshot the SDK persisted, which can lag behind graph execution.
A reusable parent needs both a stored response and its restorable checkpoint.

### Recovery flow

The Agent Server SDK schedules recovery for stored background work when
`resilient_background=True`. Response/task storage and graph checkpoints must
survive the restart. See [restart recovery](responses.md#restart-recovery) for
the shared setup.

The starting checkpoint is the main difference when recovery has no saved
graph progress:

| Persisted state | Ordinary recovery | Branch-child recovery |
| --- | --- | --- |
| No graph progress | Replay the original input through ordinary continuation; a root reuses its existing thread. | Reload and validate the parent's exact checkpoint, then replay the child's input. |
| Valid progress for the current response | Resume that checkpoint with input `None`, preserving pending writes. | Use the same resume path; the child's progress takes precedence over its parent, so no parent lookup is needed. |

Both paths start the output stream from the persisted response snapshot and
finalize an already finished graph from its saved state. Malformed, unavailable,
mismatched, or foreign progress fails recovery. Work after the last saved
snapshot, including external effects and response finalization, can run again.

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

### Follow-up

The remaining work is:

- Align ordinary continuation with branching for requests using
  `previous_response_id`: select the referenced response's checkpoint and
  skip all reads and writes to `conversation_chain_store`. Requests using
  `conversation` continue to use the conversation-chain checkpoint.
- Validate parent outputs for unanswered tool calls as well as checking graph
  completion.
- Reject requests combining `previous_response_id` and `conversation` with
  HTTP 400.
- Isolate top-level `instructions` to the current request across checkpoint
  restoration, while preserving explicit system and developer message items.

Branching from paused parents requires a separate design for isolating tool
replies, approval decisions, and resume state between children.

## 3. Use cases

### Server: enable branching

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

### Client: create two branches

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
