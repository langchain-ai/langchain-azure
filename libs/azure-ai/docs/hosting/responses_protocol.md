# Responses protocol: current behavior

This document describes `ResponsesHostServer` on `main` at
[`8fabe31`](https://github.com/langchain-ai/langchain-azure/commit/8fabe31),
reviewed on 2026-11-04. It covers the default host and converters. Subclasses
can change these behaviors. The hosting API is experimental.

The checked-in lockfile uses Agent Server Core `2.3.0b2`, Responses `2.4.0b2`,
and Invocations `1.3.0b1`. Runtime-owned behavior below was checked against
those versions. See also [Invocations protocol](invocations_protocol.md).

**Legend:** ✓ supported under the conditions in Notes; × unsupported.
`resilient_background` and `steerable_conversations` are server options;
`store`, `background`, and `stream` are request fields. All default to false
except request `store`, which defaults to true.

## Features

| Feature | Supported | Notes |
| --- | :---: | --- |
| Create a response | ✓ | `POST /responses`; `prefix` can add a route prefix such as `/v1`. |
| JSON response | ✓ | `stream=false` returns a Responses envelope with an `output` array. The graph still runs through `astream` internally. |
| SSE response | ✓ | `stream=true` emits Responses lifecycle and output events. |
| Background execution | ✓ | `background=true` requires `store=true`. Works with JSON or SSE. Crash recovery has additional requirements below. |
| Retrieve, delete, or cancel a response | ✓ | Agent Server owns `GET /responses/{id}`, `DELETE /responses/{id}`, and `POST /responses/{id}/cancel`. Retrieval requires a stored response; cancellation is for background work. Deleting a response does not delete graph checkpoints. |
| Reconnect to a stored stream | ✓ | Retrieve with `stream=true` and `starting_after`. Replay retention and snapshot recovery belong to Agent Server; clients must handle a recovered `response.in_progress` snapshot reset. |
| Conversation history without a checkpointer | ✓ | The response provider supplies prior history; the host converts history plus current input to graph messages. |
| Conversation state with a checkpointer | ✓ | LangGraph restores state; the host supplies only current input. The conversation chain store records the checkpoint for later turns. |
| Exact checkpoint branching | ✓ | Opt in with `enable_response_branching=True`; see the branching matrix below. |
| Steering an active conversation | ✓ | `steerable_conversations=True`; queues the next turn and signals the active turn to stop. Uses a linear chain and cannot be combined with checkpoint branching. |
| Human-in-the-loop (HITL) | ✓ | A checkpointer is required for resumable interrupts. Both JSON and SSE expose paired `function_call` and `mcp_approval_request` items. |
| Text, message roles, and instructions | ✓ | User, assistant, system, and developer messages are converted; developer becomes a system message. Current `instructions` becomes a `SystemMessage`. |
| Image and file input | ✓ | Responses-native `input_image` and `input_file` blocks are preserved. The configured model must accept them, for example through `use_responses_api=True`. The host does not fetch or parse files itself. |
| Tool calls and results in output | ✓ | Graph tool calls become `function_call`; `ToolMessage` results become `function_call_output`. Text and Responses-native image/file tool results are preserved. Tool arguments are published from completed node updates. |
| Reasoning summaries | ✓ | Converts model-provided `reasoning.summary` blocks. The model must be configured to emit summaries. |
| Citations | ✓ | Preserves valid Responses-native text annotations. Unsupported annotation shapes are omitted; answer text remains. |
| Token usage | ✓ | Aggregates LangChain `usage_metadata` into response usage, including available cache and reasoning counts. Missing metadata is not estimated. |
| Select the graph's model or tools through request fields | × | The default converter does not apply request `model`, `tools`, or sampling settings to the graph. Configure the graph itself. |
| Arbitrary graph state or custom LangGraph stream events | × | The default schema requires `messages`; it consumes messages, updates, and checkpoint events. Override host hooks for a different mapping. |
| CLI startup from `langgraph.json` | ✓ | `python -m langchain_azure_ai.agents.hosting.run --protocol responses`; repeat `--option NAME=VALUE` for `ResponsesServerOptions`. Branching is a host constructor argument and is not exposed by `--option`. |

## Conversation and branching combinations

The host automatically chooses one state source from the graph's checkpointer.
`previous_response_id` alone does not enable exact checkpoint branching.

| Graph / host configuration | Request linkage | Supported | Starting state and notes |
| --- | --- | :---: | --- |
| No checkpointer; branching off | None | ✓ | Current input only. |
| No checkpointer; branching off | `previous_response_id` or `conversation` | ✓ | Provider-resolved transcript plus current input. This restores message history, not graph runtime state. |
| Checkpointer; branching off | None | ✓ | A new response-root thread; no prior conversation state. |
| Checkpointer; branching off | `conversation.id` | ✓ | A user-scoped conversation thread and its saved conversation-chain checkpoint, when present. This is linear continuation. |
| Checkpointer; branching off; steering on | `previous_response_id` | ✓ | A linear response chain and its committed checkpoint. Agent Server requires the latest parent and rejects forks. |
| Checkpointer; branching off; steering off | `previous_response_id` | ✓ | Resolves the parent's conversation or root thread. Without a saved reference for the current chain, LangGraph selects the latest checkpoint in that thread. An older response ID does not select its exact checkpoint. |
| History-preserving async checkpointer; branching on; steering off | `previous_response_id` | ✓ | The named completed response's exact `(thread_id, checkpoint_id)`. Children share the thread and keep separate checkpoints. |
| Branching on | `conversation.id`, no `previous_response_id` | ✓ | Ordinary linear conversation path; enabling branching does not change this request. |
| Branching on | No linkage | ✓ | A new root response. |
| No checkpointer; branching on | Any | × | Constructor raises `ValueError`. |
| Branching on; steering on | Any | × | Constructor raises `ValueError`. |
| Branching on; external `app` supplied | Any | × | Constructor raises `ValueError`. Without branching, an external app is supported and owns host-level configuration. |

An explicit conversation uses its ID as the logical graph thread. A response
chain resolves its conversation or root response ID. When the platform supplies
`user_id_key`, the host prefixes the thread with a hash of that key. Backend
user isolation is a separate storage setting.

### Branching details

Set `enable_response_branching=True` on the Python host constructor. The saver
must retain history and implement asynchronous `aget_tuple`, `aput`, and
`aput_writes`; a synchronous-only saver is rejected. An in-memory saver can
exercise branching locally; durable recovery also requires persistent storage.

| Branching combination | Supported | Notes |
| --- | :---: | --- |
| Two children of the same completed parent | ✓ | Both start from the parent. If A is the parent, B sees A+B and C sees A+C. Sequential and concurrent siblings are covered by tests. |
| A child used as a later parent | ✓ | A descendant of B sees A+B+its own input. |
| JSON or SSE | ✓ | Both use the same parent selection. |
| Parent created through an explicit conversation | ✓ | Select its stored response ID. The child uses that response's checkpoint and does not advance the conversation-chain head. |
| Parent from another user partition | × | The host checks the stored checkpoint thread namespace against the current platform user namespace. |
| Missing response, missing checkpoint reference, or deleted checkpoint | × | The request fails; it does not start a fresh root. Both the response record and graph history must remain available. |
| Parent response is completed but its graph is paused | × | Pending tasks, next nodes, or interrupts make the parent ineligible. A completed protocol response does not imply a finished graph. |
| Resume a paused parent using `previous_response_id` while branching is on | × | It reaches the branch eligibility check before HITL resume. For workflows that pause, keep branching disabled and use ordinary conversation continuation. |
| Store the child response | ✓ | Request `store` is honored. With `store=false`, the child has no retained response record for later response-ID branching; graph checkpoint writes still occur. |
| Background recovery of a child | ✓ | Requires the recovery combination below. The child's own saved progress takes precedence; before it has progress, recovery selects the original parent again. |
| Fresh request instructions isolated from parent instructions | × | Prior system messages can remain in checkpoint state. The option does not remove them. |
| Reject `conversation` together with `previous_response_id` | × | The current branch selector prioritizes `previous_response_id`. This combination is not validated as mutually exclusive by the checked runtime. Send one linkage field. |
| Validate every unanswered external tool call on the parent | × | The host checks graph completion and pause state; it has no separate parent-output eligibility check for unanswered external calls. |

Branch-parent eligibility failures inside the handler occur after
`response.created` and become `response.failed` with `server_error`. They do not
all become an HTTP 4xx before execution; Agent Server can reject malformed or
unavailable references earlier.

The existing [branching design](response_branching.md) describes a target
contract and further work. Use the tables here for the behavior implemented
at this revision.

## Persistence and recovery combinations

| Request `store` | Request `background` | Server `resilient_background` | Request accepted | Handler resumes after restart | Notes |
| :---: | :---: | :---: | :---: | :---: | --- |
| ✓ | × | × | ✓ | × | Foreground request. |
| ✓ | × | ✓ | ✓ | × | Enabling resilience does not make foreground Responses handlers recoverable. |
| ✓ | ✓ | × | ✓ | × | Background execution and retrieval; no handler crash recovery. |
| ✓ | ✓ | ✓ | ✓ | ✓ | Requires a graph checkpointer and persistent protocol, chain, graph, and task state. |
| × | × | Either | ✓ | × | No stored response. Graph checkpoints and conversation-chain writes are independent of request `store`. |
| × | ✓ | Either | × | × | The checked Agent Server version rejects `background=true` with `store=false`. |

`stream` can be true or false for each accepted row. Steering can be enabled
independently of resilience, subject to the branching restriction. Creating the
default app with `resilient_background=True` and no graph checkpointer raises
`ValueError`, regardless of the eventual request flags. For stored
work that cannot resume, an enabled Agent Server task runtime marks an
interrupted response failed on restart. With both server options off, no
restart task performs that cleanup. The upstream
[resilience contract](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-responses/docs/resilience-contract.md)
defines the framework lifecycle; the matrix above also includes the checked
runtime's request validation.

Three stores have distinct responsibilities:

| State | Owner / default | Why it is needed |
| --- | --- | --- |
| Response envelope, items, status, and current-turn checkpoint reference | Agent Server `store`: `FileResponseStore` locally; `FoundryStorageProvider` in a configured Foundry host | Retrieval and the current response's recovery snapshot. |
| Committed conversation checkpoint reference | Host `conversation_chain_store`: `FoundryConversationChainStore` | Selects the checkpoint for the next linear turn. Custom implementations must persist dictionaries and propagate storage errors. |
| Graph state and pending writes | The graph's configured LangGraph checkpointer | Restores execution and application state. An in-memory saver is lost on restart. |

At each graph checkpoint, the host records its reference with the Responses
output snapshot. Recovery seeds output from that snapshot and resumes the
graph with `input=None` at the recorded checkpoint. If the current response
has no saved graph progress, it replays the original request through normal
input/HITL handling. Malformed saved checkpoint metadata fails the response.
Completed graph nodes or external effects can run again across an uncommitted
boundary; tools still need replay-safe behavior. See the
[recovery design](responses_resilient_design.md) for the write sequence.

## HITL combinations

The client sends resume items in `input`. The graph's interrupt ID is the
`function_call.call_id`; the paired approval item has its own generated
`mcpr_*` ID. Answer the approval with that item's ID.

| Input / pending state | Supported | Behavior |
| --- | :---: | --- |
| Matching `function_call_output` | ✓ | Decodes a rich `resume`, `update`, or `goto` command. It takes precedence over an approval for the same interrupt. |
| Custom interrupt + `mcp_approval_response`, `approve=true` | ✓ | Resumes with the original interrupt value. |
| Custom interrupt + `approve=false` | × | Fails with `interrupt_rejected`; leaves the graph paused. |
| Recognized `HumanInTheLoopMiddleware` payload + approval/rejection | ✓ | Expands the boolean choice to ordered action decisions, if every action permits that decision. Use rich output for edited arguments or custom responses. |
| Conflicting approvals or a disallowed middleware decision | × | Rejects the whole request before resuming any pending interrupt. |
| Answer only some parallel interrupts | ✓ | Resumes the selected interrupts; only outstanding interrupts are emitted again. |
| Plain input or unmatched resume while paused | ✓ | Re-emits pending items and completes the response without driving the graph with fresh input. |

## Completion, cancellation, and errors

| Cause | Handler output | Notes |
| --- | --- | --- |
| Graph finishes | `response.completed` | Final output and available usage are returned. |
| Graph pauses at an interrupt | `response.completed` plus interrupt items | The protocol turn is complete; graph execution remains paused. |
| Client cancellation signal | `response.failed` with code `cancelled` | The handler distinguishes cancellation from steering. Agent Server owns the cancel endpoint and persisted lifecycle. |
| Steering signal | `response.completed` | Preserves checkpointed partial output before the queued turn starts. |
| Shutdown signal | Exit for recovery | Resumption depends on the persistence/recovery matrix above. |
| Graph or handler exception | `response.failed` with code `server_error` | Public message: `An internal error occurred.` Details remain in server logs. |

Graph cleanup must finish before the handler advances the conversation-chain
checkpoint. A graph-close timeout leaves that reference unchanged; during
steering it fails the turn instead of completing it with unsettled writes.

## Implementation and verification references

- [Host and configuration](../../langchain_azure_ai/agents/hosting/_responses_host.py),
  [stream conversion](../../langchain_azure_ai/agents/hosting/_converters/_stream.py),
  [input conversion](../../langchain_azure_ai/agents/hosting/_converters/_request.py),
  and [HITL conversion](../../langchain_azure_ai/agents/hosting/_converters/_hitl.py).
- [Host tests](../../tests/unit_tests/agents/hosting/test_responses_host.py),
  [branching tests](../../tests/unit_tests/agents/hosting/test_response_branching.py),
  [HITL tests](../../tests/unit_tests/agents/hosting/hitl/), and
  [local/remote crash harness](../../tests/e2e_tests/agents/hosting/responses_resilience/README.md).
- [Hosting samples](../../../../samples/hosting/langgraph-hosted-agents/README.md):
  workflows, files, tools, HITL, resilience, and observability.

These are source and test references, not a claim that every combination was
executed against a live Foundry deployment during this documentation review.
