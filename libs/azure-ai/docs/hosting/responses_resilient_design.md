# Foundry-based Resilient LangGraph Agent (Responses API)

## Background

### Problem

Long-running agent runs face two independent failures:

1. The client can disconnect while the turn is still running; the client must be able
to reconnect and retrieve the same response.
2. The server process can
crash or restart; the agent must recover its execution state and continue the remaining work without any internal inconsistencies.

### The OpenAI Responses API

The Responses API addresses the client disconnection problem with
[background mode](https://developers.openai.com/api/docs/guides/background): a
client creates a response with `background=true` and polls the response. With
`stream=true`, the client can also reconnect with `starting_after`. However, it
doesn't explicitly define how the server crash recovery works and how the server rolls back  events and chunks client already receives.

### How Foundry Extends the Responses API

Recovery of the server host process is a separate contract defined by Foundry
and the Agent Server SDK. In its
[resilience matrix](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-responses/docs/resilience-contract.md#the-matrix),
full crash recovery applies when the request has `store=true` and
`background=true` and the server opts in with `resilient_background=True`; the
framework then re-invokes the application **handler**, but the handler
still owns application progress and side-effect safety.

### What We Do

Our SDK already owns the **handler** and handles two-way communication
between the Responses contract and LangGraph events. LangGraph stores the
durable execution state through its checkpoint saver. This design composes the
two recovery boundaries so the client-visible response and LangGraph execution
resume from the same committed point after a restart. See the
[resilient LangGraph sample](../../../../samples/hosting/langgraph-hosted-agents/responses/10_resilient/README.md)
to understand a real-world, user-visible flow.

What happens without our work?

With `resilient_background=True`, Agent Server provides the framework half of
recovery: it durably records the request, response, and events, then re-invokes
the handler with the same `response_id` and identical request input. If we do
nothing recovery-specific, the SDK will consider this a brand-new turn, rerun
the whole turn, and re-stream it after a `response.in_progress` snapshot reset.
The original partial output are lost and **the final response can still be coherent,
but upstream side effects may run twice**.

## Durable State Model

There are three independent storage responsibilities. The first two correspond
to the `ResponsesHostServer` constructor parameters `store` and
`conversation_chain_store`; the third belongs to the compiled LangGraph.

| Store | Scope | Key | Content | Authority |
| --- | --- | --- | --- | --- |
| Response store (`store`) | Single turn | `response_id` | Agent Server-owned response, input, output, status, events, and current-turn checkpoint metadata. Agent Server defaults to `FileResponseStore` locally and `FoundryStorageProvider` on Foundry. | Source of truth for the client-visible response and recovery metadata of the current turn |
| Conversation chain store (`conversation_chain_store`) | Conversation | `(conversation_chain_id, "langgraph_checkpoint")` | Latest completed-turn `{thread_id, checkpoint_id}`. The default `FoundryConversationChainStore` uses one `FoundryStateStore` namespace per conversation chain. | Source of truth for which committed checkpoint starts the next turn |
| LangGraph checkpoint saver | Single turn + conversation | `thread_id, checkpoint_ns, checkpoint_id` | Full graph state and execution progress | Source of truth for graph execution state; references in the other stores point here |

The normal conversation-chain pattern is a lookup by conversation namespace and
string key. `get(conversation_chain_id, key)` returns a dictionary or `None`;
`set(conversation_chain_id, key, data)` atomically replaces the dictionary.
The backend owns serialization. The hosting code defines the
`langgraph_checkpoint` dictionary and converts it to and from `CheckpointRef` at
the host boundary.

Conversation chain storage does not replace the response store or LangGraph
checkpointer. Without the conversation checkpoint reference, the checkpointer
may contain dangling work that advanced beyond the last committed Responses
boundary, and a later turn would not know which checkpoint is authoritative.

### Non-stored Turns

For `store=false`, the host reads the committed conversation checkpoint into a
request-local in-memory branch. Graph checkpoints and pending writes stay in
that branch; the configured LangGraph saver and the conversation checkpoint
reference are not updated. The host does not emit SDK recovery checkpoint
events for the turn. This applies to both streaming and non-streaming requests.

Pending-write indices are preserved for Foundry, in-memory, and SQLite savers.
Other savers must expose original indices to import ordinary pending writes;
otherwise the request fails explicitly rather than risking duplicate outputs.

Built-in Foundry conversation and checkpoint reads never create a state store.
If a store does not exist, the read returns no saved state without creating an
empty store or its metadata. Only write operations create missing stores.
Reads of existing stores preserve their properties and stored items. Storage
failures other than a missing store continue to propagate.

**Crash recovery is not supported for `store=false`**, including when the server
enables `resilient_background`. Request-local checkpoints only support normal
conversation reads and interrupt handling within the isolated branch; they are
not recovery checkpoints and do not survive a process crash.

- The host never restores a persisted response snapshot for a non-stored turn.
- If a non-stored turn is invoked with `context.is_recovery=True`, it emits
  `response.failed` with code `unsupported_recovery` before reading graph state
  or executing graph nodes. It does not replay the request input.
- On graceful shutdown, a non-stored turn emits `response.failed` with code
  `server_shutdown`. It never calls `context.exit_for_recovery()` or leaves the
  turn waiting for execution in the next process lifetime.
- After a crash, the client must start a new request. A retry does not resume the
  discarded branch; tools with external side effects must remain replay-safe.

The branch is discarded after the request, including newly created interrupts.
A non-stored resume of an existing interrupt leaves the saved interrupt pending.
This flag does not roll back tool side effects or disable long-term store writes
or tracing. `store=true` and the default behavior retain the durable flow below;
Agent Server SDK 2.3.0 rejects `background=true` with `store=false`.

### State Store client lifetime

Each Responses handler owns a `StateStoreClientFactory` from config resolution
through the final conversation-chain write. Clients are separate by consumer,
store name and user; hosted clients share one connection pool. Store existence is
refreshed on reuse. Reads never create missing stores. No checkpoint values or
descriptors are cached; isolation and TTL remain creation options.

The factory and request features are bound for each iterator advance or close,
allowing iteration across Tasks. Reuse stays within one handler and event loop;
recovery creates a new factory. Standalone operations and custom transports,
pipelines or sessions retain their existing lifetime.

Cleanup waits for active storage operations, cancels independent borrowers on
timeout, then closes clients and the pool. Handler Tasks are protected from this
cancellation. Iterator closes, borrower drain/cancellation waits and resource
closes each have a five-second budget, with no shared total deadline or limit on
graph runtime. Cleanup preserves primary errors, recovery signals and generator
exit; caller cancellation takes precedence. Secondary cleanup errors are logged;
normal-exit cleanup failures propagate.

A graph-close timeout during normal iteration leaves the conversation-chain
checkpoint unchanged. Shutdown exits for recovery for stored turns; non-stored
turns fail with `server_shutdown`. Client cancellation reports `cancelled`, and
steering reports `server_error`. Explicit `aclose()` preserves generator exit and
emits no terminal events, even if cleanup times out.
Execution errors report `server_error`. All `server_error` messages are
`An internal error occurred.`; internal details stay in logs.

### Assumptions

The recovery behavior depends on these assumptions to work properly:

- All three stores survive host-process restarts.
- A successful `stream.checkpoint()` atomically persists output and its
  LangGraph checkpoint reference. Provider failures are logged and swallowed.
- Foundry State Store guarantees cross-turn persistence.
- Observing `response.created` on the client proves that Agent Server created
  and persisted the response envelope. See the
  [response persistence model](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-responses/docs/responses-resilience-spec.md#82--the-recovery-model)
  and
  [streaming sub-contract](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-responses/docs/resilience-contract.md#streaming-sub-contract).
- Most completed nodes are not replayed because recovery resumes from the
  last checkpoint. Corner cases are a node
  crashes before its LangGraph checkpoint commits, and a completed node
  whose checkpoint commits before the matching `stream.checkpoint()` succeeds.
  These node executions should be side-effect-free or idempotent.

## Crash Recovery

The recovery flow below applies to stored background responses with
`resilient_background=True`. Non-stored turns are excluded.

### Important Events

After Agent Server invokes or re-invokes our handler, the durable writes occur
in this order:

| Event | Write |
| --- | --- |
| E1. `stream.emit_created()` | Agent Server writes the current request's input and metadata. |
| E2. LangGraph commits a checkpoint | LangGraph writes the state to its checkpoint saver and emits `stream_mode="checkpoints"` events. |
| E3. Store the checkpoint reference and yield `stream.checkpoint()` | `TaskStorageManager.store_checkpoint_ref()` stores the newly committed checkpoint reference in response `internal_metadata`. `StreamConverter.checkpoint()` then causes Agent Server to write the response output and checkpoint reference together. E2 and E3 repeat for each superstep. |
| E4. Call `ConversationChainStoreProtocol.set()` | After the graph finishes, the handler saves the final `{thread_id, checkpoint_id}` for the next turn. |
| E5. Yield `stream.emit_completed()` | Agent Server writes the terminal response. |

### Crash Windows

| # | Crash between | Response store | LangGraph checkpoint saver | Conversation chain storage | Recovery behavior |
| ---: | --- | --- | --- | --- | --- |
| 1 | Before E1 | None. | Previous turn checkpoint, or none for a new conversation. | Previous turn checkpoint reference, or none for a new conversation. | The client retries the original request. |
| 2 | E1 -> E2 | Input and metadata. | Same as above. | Same as above. | Agent Server re-invokes our handler with the same request input. Our handler replays that input and runs the first superstep. |
| 3 | E2 -> E3 (superstep 1) | Same as above. | Previous turn checkpoint and the checkpoint committed by superstep 1. | Same as above | The response snapshot remains the recovery authority. Our handler replays the request input from the previous turn checkpoint and reproduces the first superstep. The original checkpoint committed by superstep 1 is left dangling and is deleted when TTL expires. |
| 4 | E3 -> E2 (superstep 1 -> 2) | Input, output through superstep 1, the checkpoint reference for superstep 1, and metadata. | Checkpoints through superstep 1. | Same as above. | Our handler calls `graph.astream(input=None)` from that checkpoint and reruns the interrupted superstep. Nodes and external effects must be replay-safe or idempotent. |
| 5 | E2 -> E3 (superstep 2+) | Same as above. | Checkpoints through the newly committed superstep 2+. | Same as above. | The Response store remains the recovery authority. Our handler resumes from its checkpoint reference and reproduces the newly committed superstep. |
| 6 | E3 -> E4 | Input, final output, final checkpoint reference, and metadata. | Checkpoints through the final checkpoint. | Same as above. | Our handler resumes from the final checkpoint and writes its reference to Foundry State Store. |
| 7 | E4 -> E5 | Same as above. | Same as above. | Final checkpoint reference. | Our handler resumes from the final checkpoint, idempotently writes its reference to Foundry State Store, and yields `stream.emit_completed()`. |
| 8 | After E5 | Input, final output (including the `completed` chunk), final checkpoint reference, and metadata. | Save as above. | Same as above. | Agent Server treats the persisted terminal response as authoritative and returns it to clients through response retrieval or stream replay. |
