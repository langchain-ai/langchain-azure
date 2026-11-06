# Responses hosting overview

## 1. Interface

[`ResponsesHostServer`](https://reference.langchain.com/python/langchain-azure-ai/agents/hosting/_responses_host/ResponsesHostServer) hosts a compiled
[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) graph through the [Responses protocol](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/langchain-hosted-agents#responses-protocol).
The [Agent Server SDK](https://learn.microsoft.com/en-us/python/api/azure-ai-agentserver-core/azure.ai.agentserver.core?view=azure-python) provides the protocol; this host handles
graph input, output, and state. See also
[Invocations hosting](invocations.md).

### Dependencies

Install `langchain-azure-ai[hosting]`. It includes
[LangChain](https://docs.langchain.com/oss/python/langchain/overview), LangGraph,
and the Agent Server SDK. Dependency ranges are declared in
[the package manifest](../../pyproject.toml).

[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) is optional
and uses the same LangGraph hosting interface.

### Constructor

| Constructor parameter | Default | Purpose |
| --- | --- | --- |
| `graph` | Required | The compiled graph to serve. |
| `app` | `None` | Attach to an existing Agent Server app, or create one when omitted. |
| `options` | `None` | Use default [`ResponsesServerOptions`](https://learn.microsoft.com/en-us/python/api/azure-ai-agentserver-responses/azure.ai.agentserver.responses.responsesserveroptions?view=azure-python): `resilient_background=False` and `steerable_conversations=False`. |
| `store` | `None` | If not specified, uses [`FileResponseStore`](https://learn.microsoft.com/en-us/python/api/azure-ai-agentserver-responses/azure.ai.agentserver.responses.fileresponsestore?view=azure-python) (JSON files) locally, or [`FoundryStorageProvider`](https://learn.microsoft.com/en-us/python/api/azure-ai-agentserver-responses/azure.ai.agentserver.responses.foundrystorageprovider?view=azure-python) (Foundry storage API) when hosted in [Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/hosted-agents). |
| `conversation_chain_store` | `None` | If not specified, uses [`FoundryConversationChainStore`](https://reference.langchain.com/python/langchain-azure-ai/agents/hosting/_responses/conversation_chain_store/FoundryConversationChainStore), backed by the [durable state store](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/agent-state-store), to retain the [checkpoint](https://docs.langchain.com/oss/python/langgraph/persistence#checkpoints) reference for the next linear turn. |
| `enable_response_branching` | `False` | Continue from a named response's exact graph checkpoint. |
| `prefix` | Empty | Add a prefix to the protocol routes. |
| `applicationinsights_connection_string` / `graceful_shutdown_timeout` | `None` | Configure telemetry / graceful shutdown on the created app. |

> [!NOTE]
> When supplying `app`, configure its host-level settings, including response
> storage, on that app. Supply the graph's [checkpointer](https://docs.langchain.com/oss/python/langgraph/checkpointers) when compiling it.
> The constructor's `store` provider and the request's `store` retention flag
> are separate controls.

Start serving with `run()` or `run_async()`; `.app` and `.graph` expose the
underlying server and hosted graph.

> [!TIP]
> The [CLI](../../langchain_azure_ai/agents/hosting/run.py) supports
> `--protocol responses` and `--option NAME=VALUE`. Configure other constructor
> parameters, including `app` and [branching](https://docs.langchain.com/oss/python/langgraph/use-time-travel), through Python.

## 2. Calling modes and input/output

Request defaults are `store=true`, `stream=false`, and `background=false`.
✓ means supported under the stated conditions; × means unsupported.

Create, retrieve, cancel, delete, input-item listing, and [server-sent events (SSE)](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses) formatting follow
[Agent Server Responses](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-responses/README.md#protocol-endpoints).
A create request returns JSON or SSE according to `stream`. Background execution
requires `store=true`; stored background streams support reconnecting with
`starting_after`. Deleting a response leaves graph checkpoints intact.

The default graph schema requires `messages`. Text, instructions, tool items,
and Responses-native image/file input are supported with a compatible model.
LangGraph message and update streams produce assistant text, tool calls/results,
and model-provided reasoning summaries, citations, and usage. Custom graph state
and custom stream events have no default mapping. Models and tools are configured on
the graph; request `model` and `tools` do not reconfigure it.

## 3. Conversation and state

> [!IMPORTANT]
> When a graph has a checkpointer, requests must use `store=true` (the default).
> `store=false` returns HTTP 400 before streaming or graph execution. Without a
> checkpointer, foreground requests can use `store=false`.

A LangGraph checkpointer determines the conversation's state source:

| Graph configuration | History used on later turns | Input sent to the graph |
| --- | --- | --- |
| Without a checkpointer | Responses history resolved by the provider | History plus current input. |
| With a checkpointer | Saved LangGraph state | Current input only. |

`conversation.id` identifies a conversation [thread](https://docs.langchain.com/oss/python/langgraph/persistence#threads). For `previous_response_id`,
the host resolves the parent's conversation or root thread. Threads are scoped
to the platform user when that identity is available. An in-memory saver keeps
state within one process; a persistent saver is needed across restarts.

Ordinary checkpointed continuation uses the saved conversation-chain checkpoint
when available. Without one, LangGraph selects the latest checkpoint in the
thread. Exact selection of an older response's checkpoint requires branching.

### Storage choices

The repository provides the following implementations for different state roles:

| Role | Implementation | Package / backend |
| --- | --- | --- |
| Conversation-chain references | `FoundryConversationChainStore` | `langchain-azure-ai`; durable state store. |
| Graph checkpoints | [`FoundryCheckpointSaver`](https://reference.langchain.com/python/langchain-azure-ai/agents/hosting/_foundry_checkpoint_saver/FoundryCheckpointSaver) | `langchain-azure-ai`; durable state store. Requires hosted request context. |
| Graph checkpoints | [`CosmosDBSaver`](https://reference.langchain.com/python/langchain-azure-cosmosdb/aio/_langgraph_checkpoint_store/CosmosDBSaver) | `langchain-azure-cosmosdb`; [Azure Cosmos DB for NoSQL](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/overview). |

Supply a checkpointer through `compile(checkpointer=...)`; the host does not add
one. Both hosts require the async `CosmosDBSaver` rather than
[`CosmosDBSaverSync`](https://reference.langchain.com/python/langchain-azure-cosmosdb/_langgraph_checkpoint_store/CosmosDBSaverSync). Cosmos DB implementations require installing
`langchain-azure-cosmosdb` separately.

### Steering

[Steering](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-responses/docs/resilient-responses-developer-guide.md) uses `steerable_conversations=True` to queue the next input
and stop the active turn. It enforces a linear chain and rejects a supplied
parent that is no longer the latest response. A superseded response normally
completes with partial output.

### Branching

Branching uses `enable_response_branching=True` with `previous_response_id`.
It restores the named parent's checkpoint. Siblings can run sequentially or
concurrently: children B and C of A see A+B and A+C respectively. Each child
can become a new parent.

> [!NOTE]
> Branching requires `previous_response_id`; omit `conversation` on a branch
> request. Using only `conversation.id` continues the conversation normally,
> even when branching is enabled. Each branch saves its own response and graph
> checkpoint without updating `conversation_chain_store`.

| Combination | Supported | Notes |
| --- | :---: | --- |
| Branching + async, history-preserving checkpointer | ✓ | The parent response and its checkpoint must remain available to the current user. |
| Branching without a checkpointer | × | Rejected at host construction. |
| Branching + `steerable_conversations=True` | × | Currently unsupported. Disable steering to use branching. |
| Branching from a paused graph | × | Includes [human-in-the-loop (HITL)](https://docs.langchain.com/oss/python/langgraph/interrupts) resume through `previous_response_id` in branching mode. Use ordinary continuation for workflows that pause. |
| Branching + external `app` | × | Branching requires a host-created app. |
| Branching + stored background recovery | ✓ | Subject to the recovery requirements below. |

Parent system messages can remain in checkpoint state. Missing or inaccessible
parents return HTTP 404; invalid branch parents or checkpoints return HTTP 400
before streaming. Unexpected storage errors during these checks return HTTP 500.
Errors during graph execution return a response with
`status="failed"`; streaming clients receive a `response.failed` event.

## 4. Interrupts and recovery

### Human-in-the-loop

LangGraph [`interrupt()`](https://docs.langchain.com/oss/python/langgraph/interrupts) pauses a checkpointed graph. The host exposes two
ways to answer the same pause; use either path. The response can be `completed`
while the graph remains paused.

| Resume path | Output item | Reply item |
| --- | --- | --- |
| Rich resume value or command | `function_call` | `function_call_output` |
| Boolean approval | `mcp_approval_request` | `mcp_approval_response` |

For one pause, routing fields from the emitted SSE events look like this:

```json
[
  {"type": "response.output_item.done",
   "item": {"type": "function_call", "call_id": "interrupt-1"}},
  {"type": "response.output_item.done",
   "item": {"type": "mcp_approval_request", "id": "mcpr_example", "server_label": "langgraph"}}
]
```

Continue the same conversation with branching disabled. To provide a resume
value, send a `function_call_output` using the emitted `call_id`:

```json
{
  "conversation": {"id": "<same-conversation-id>"},
  "input": [{"type": "function_call_output", "call_id": "interrupt-1",
             "output": "{\"resume\":\"continue\"}"}]
}
```

For a boolean decision, send an `mcp_approval_response` using the approval
item's `id` as `approval_request_id`:

```json
{
  "conversation": {"id": "<same-conversation-id>"},
  "input": [{"type": "mcp_approval_response",
             "approval_request_id": "mcpr_example", "approve": true}]
}
```

The host converts the answer into a LangGraph resume command. For custom
interrupts, positive approval resumes with the original interrupt value;
negative approval fails the request and leaves it paused. [`HumanInTheLoopMiddleware`](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
supports allowed per-action approval/rejection decisions. Conflicting or
disallowed decisions fail the request. Independent parallel interrupts can be
answered separately; unmatched input re-emits outstanding pauses.

> [!NOTE]
> If both reply types answer the same interrupt, `function_call_output` takes
> precedence. The `call_id` and `approval_request_id` come from different output
> fields; copy each from the corresponding emitted item.

### Restart recovery

| Scenario | Restart recovery | Conditions / behavior |
| --- | :---: | --- |
| Foreground request | × | The resilience option does not resume a foreground Responses handler. |
| Stored background request; resilience off | × | Background execution alone provides no handler crash recovery. |
| Stored background request; resilience on | ✓ | Requires persistent response/task storage, conversation-chain state, and graph checkpoints. The host requires a checkpointer when enabling resilience. |

The response store retains protocol data, the conversation-chain store retains
the checkpoint for the next linear turn, and the graph saver retains execution
state.

Recovery uses the current response's saved graph progress, including for a
branch child. Without saved progress, it processes the original input again.

> [!WARNING]
> External effects can repeat across a recovery boundary.

See the public
[Agent Server resilience contract](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-responses/docs/resilience-contract.md)
for the protocol lifecycle.

Client cancellation fails the handler with code `cancelled`. Shutdown recovery
follows the conditions above.
