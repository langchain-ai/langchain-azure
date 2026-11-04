# Responses hosting behavior

## 1. Scope and defaults

This is a high-level summary of the default `ResponsesHostServer` on
[main at `8fabe31`](https://github.com/langchain-ai/langchain-azure/commit/8fabe31),
reviewed on 2026-11-04. Hosting is experimental. The lockfile uses Agent Server
Core `2.3.0b2` and Responses `2.4.0b2`.

Agent Server provides the Responses protocol; this host maps requests to a
LangGraph graph and converts its output. See also
[Invocations hosting](invocations_protocol.md). ✓ means supported under the
stated conditions; × means unsupported.

| Control | Default | Purpose |
| --- | --- | --- |
| Request `store` | `true` | Retain the response for later retrieval. |
| Host `store` (provider) | `None` (automatic) | Uses `FileResponseStore` locally or `FoundryStorageProvider` when hosted on Foundry; stores response records, items, status, and history. |
| Host `conversation_chain_store` | `FoundryConversationChainStore` | Retain the checkpoint reference for the next linear turn. |
| Request `stream` / `background` | `false` | Select SSE delivery / background execution. |
| Server `resilient_background` / `steerable_conversations` | `False` | Enable restart recovery / steering. |
| Host `enable_response_branching` | `False` | Continue from a named response's exact graph checkpoint. |
| Graph checkpointer | Application-supplied | Save graph state for later turns, interrupts, and recovery. |

The host's `store` selects a response provider; the request's `store` is a
retention flag. When attaching to an existing `app`, configure its response
provider on that app.

Server options use `ResponsesServerOptions`. The [CLI](../../langchain_azure_ai/agents/hosting/run.py)
supports `--protocol responses` and `--option NAME=VALUE`; branching is configured
through the Python host constructor.

## 2. Calling modes and input/output

Create, retrieve, cancel, delete, input-item listing, and SSE formatting follow
[Agent Server Responses](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-responses/README.md#protocol-endpoints).
A create request returns JSON or SSE according to `stream`. Background execution
requires `store=true`; stored background streams support reconnecting with
`starting_after`. Deleting a response leaves graph checkpoints intact.

The default graph schema requires `messages`. Text, instructions, tool items,
and Responses-native image/file input are supported with a compatible model.
LangGraph message and update streams produce assistant text, tool calls/results,
and model-provided reasoning summaries, citations, and usage. Custom graph state
and stream events need custom conversion. Models and tools are configured on
the graph; request `model` and `tools` do not reconfigure it.

## 3. Conversation and state

A LangGraph checkpointer determines the conversation's state source:

| Graph configuration | History used on later turns | Input sent to the graph |
| --- | --- | --- |
| Without a checkpointer | Responses history resolved by the provider | History plus current input. |
| With a checkpointer | Saved LangGraph state | Current input only. |

`conversation.id` identifies a conversation thread. For `previous_response_id`,
the host resolves the parent's conversation or root thread. Threads are scoped
to the platform user when that identity is available. An in-memory saver keeps
state within one process; a persistent saver is needed across restarts.

Ordinary checkpointed continuation uses the saved conversation-chain checkpoint
when available. Without one, LangGraph selects the latest checkpoint in the
thread. Exact selection of an older response's checkpoint requires branching.

### Branching and steering

Branching uses `enable_response_branching=True` with `previous_response_id`.
It restores the named parent's checkpoint. Siblings can run sequentially or
concurrently: children B and C of A see A+B and A+C respectively. Each child
can become a new parent; branch children leave the shared conversation head
unchanged.

| Combination | Supported | Notes |
| --- | :---: | --- |
| Branching + async, history-preserving checkpointer | ✓ | The parent response and its checkpoint must remain available to the current user. |
| Branching without a checkpointer | × | Rejected at host construction. |
| Branching + steering | × | Steering is linear: it queues the next input and stops the active turn. |
| Branching from a paused graph | × | Includes HITL resume through `previous_response_id` in branching mode. Use ordinary continuation for workflows that pause. |
| Branching + external `app` | × | Branching requires a host-created app. |
| Branching + stored background recovery | ✓ | Subject to the recovery requirements below. |

Steering enforces a linear chain and rejects a supplied parent that is no longer
the latest response. Enabling branching leaves requests using only `conversation.id`
on the ordinary continuation path. Send one linkage field: the current branch
path prioritizes `previous_response_id` when both are supplied.

Request `store=false` leaves a child unavailable for later response-ID branching
while graph writes still occur. Parent system messages can remain in checkpoint
state. Branch eligibility failures inside the handler produce a failed response.

## 4. Interrupts and recovery

### Human-in-the-loop

LangGraph `interrupt()` pauses a checkpointed graph. The host exposes each pause
as paired `function_call` and `mcp_approval_request` output items. The response
can be `completed` while the graph remains paused. Send a matching
`function_call_output` or `mcp_approval_response` in the next request's `input`;
the host converts it into a LangGraph resume command.

Rich function outputs take precedence over approvals for the same interrupt.
For custom interrupts, positive approval resumes; negative approval fails the
request and leaves the graph paused. LangChain HITL middleware supports allowed per-action
approval/rejection decisions. Conflicting or disallowed decisions fail the
request. Partial parallel answers are supported; unmatched input re-emits
outstanding pauses.

### Restart recovery

| Scenario | Restart recovery | Conditions / behavior |
| --- | :---: | --- |
| Foreground request | × | The resilience option does not resume a foreground Responses handler. |
| Stored background request; resilience off | × | Background execution alone provides no handler crash recovery. |
| Stored background request; resilience on | ✓ | Requires persistent response/task storage, conversation-chain state, and graph checkpoints. The host requires a checkpointer when enabling resilience. |

The response store retains protocol data, the conversation-chain store retains
the checkpoint for the next linear turn, and the graph saver retains execution
state. Request `store=false` does not disable graph or chain-state writes.

Recovery uses the current response's saved graph progress, including for a
branch child. Without saved progress, it processes the original input again.
External effects can repeat across a recovery boundary. See the public
[Agent Server resilience contract](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-responses/docs/resilience-contract.md)
for the protocol lifecycle.

Client cancellation fails the handler with code `cancelled`; steering normally
completes it with partial output. Shutdown recovery follows the conditions above.
