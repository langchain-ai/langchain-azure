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

There are 3 stores in play, each manages different data.

| Store | Scope | Key | Content | Authority |
| --- | --- | --- | --- | --- |
| Response store | Single turn | `response_id` | Agent server owned data (responses snapshot, input snapshot, etc.) and LangGraph checkpoint referene through `internal_metadata` | Source of truth for client-visible response and the recovery metadata of the current turn |
| LangGraph checkpoint saver | Single turn + conversation | `thread_id, checkpoint_ns, checkpoint_id` (thread_id == conversation_chain_id) | Graph state | Source of truth for graph execution state; referenced by the metadata in the response store |
| Foundry State Store | Conversation | `conversation_chain_id` | Latest completed-turn `{thread_id, checkpoint_id}` used to start the next turn | Source of truth for cross-turn continuation. Without it, the checkpointer may contain dangling work that advanced beyond the last committed Response boundary and the reference hasn't reached responses store yet. This causes the responses store and langgraph state inconsistent |

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

### Important Events

After Agent Server invokes or re-invokes our handler, the durable writes occur
in this order:

| Event | Write |
| --- | --- |
| E1. `stream.emit_created()` | Agent Server writes the current request's input and metadata. |
| E2. LangGraph commits a checkpoint | LangGraph writes the state to its checkpoint saver and emits `stream_mode="checkpoints"` events. |
| E3. Store the checkpoint reference and yield `stream.checkpoint()` | `TaskStorageManager.store_checkpoint_ref()` stores the newly committed checkpoint reference in response `internal_metadata`. `StreamConverter.checkpoint()` then causes Agent Server to write the response output and checkpoint reference together. E2 and E3 repeat for each superstep. |
| E4. Call `ConversationChainStorageManager.persist_checkpoint_ref()` | After the graph finishes, the handler writes the final `{thread_id, checkpoint_id}` to Foundry State Store for the next turn. |
| E5. Yield `stream.emit_completed()` | Agent Server writes the terminal response. |

### Crash Windows

| # | Crash between | Response store | LangGraph checkpoint saver | Foundry State Store | Recovery behavior |
| ---: | --- | --- | --- | --- | --- |
| 1 | Before E1 | None. | Previous turn checkpoint, or none for a new conversation. | Previous turn checkpoint reference, or none for a new conversation. | The client retries the original request. |
| 2 | E1 -> E2 | Input and metadata. | Same as above. | Same as above. | Agent Server re-invokes our handler with the same request input. Our handler replays that input and runs the first superstep. |
| 3 | E2 -> E3 (superstep 1) | Same as above. | Previous turn checkpoint and the checkpoint committed by superstep 1. | Same as above | The response snapshot remains the recovery authority. Our handler replays the request input from the previous turn checkpoint and reproduces the first superstep. The original checkpoint committed by superstep 1 is left dangling and is deleted when TTL expires. |
| 4 | E3 -> E2 (superstep 1 -> 2) | Input, output through superstep 1, the checkpoint reference for superstep 1, and metadata. | Checkpoints through superstep 1. | Same as above. | Our handler calls `graph.astream(input=None)` from that checkpoint and reruns the interrupted superstep. Nodes and external effects must be replay-safe or idempotent. |
| 5 | E2 -> E3 (superstep 2+) | Same as above. | Checkpoints through the newly committed superstep 2+. | Same as above. | The Response store remains the recovery authority. Our handler resumes from its checkpoint reference and reproduces the newly committed superstep. |
| 6 | E3 -> E4 | Input, final output, final checkpoint reference, and metadata. | Checkpoints through the final checkpoint. | Same as above. | Our handler resumes from the final checkpoint and writes its reference to Foundry State Store. |
| 7 | E4 -> E5 | Same as above. | Same as above. | Final checkpoint reference. | Our handler resumes from the final checkpoint, idempotently writes its reference to Foundry State Store, and yields `stream.emit_completed()`. |
| 8 | After E5 | Input, final output (including the `completed` chunk), final checkpoint reference, and metadata. | Save as above. | Same as above. | Agent Server treats the persisted terminal response as authoritative and returns it to clients through response retrieval or stream replay. |
