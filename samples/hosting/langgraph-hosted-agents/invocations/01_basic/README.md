# What this sample demonstrates

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent hosted
using the **Invocations protocol** with session management. Unlike the
Responses protocol, the Invocations protocol does **not** surface
intermediate tool calls — clients receive only the final assistant text
(or its token deltas when streaming).

Multi-turn continuity is provided by a LangGraph `MemorySaver`
checkpointer: the resolved `agent_session_id` is forwarded to the graph
as `RunnableConfig.configurable.thread_id`, so each session's history
is preserved in process memory. In production, replace `MemorySaver`
with durable storage (Redis, Cosmos DB, Foundry-managed checkpoints)
so history survives restarts.

## How It Works

### Model Integration

The agent uses `langchain_openai.ChatOpenAI` pointed at the Foundry
project's `/openai/v1` endpoint, authenticated with
`DefaultAzureCredential`. The graph is a stock
`create_agent(model, tools=[], checkpointer=MemorySaver())`, so every
turn is just one chat-completion call against persisted history.

See [main.py](main.py) for the full implementation.

### Agent Hosting

The agent is hosted using
[`langchain_azure_ai.agents.hosting.LangGraphInvocationsHostServer`](../../../../libs/azure-ai/langchain_azure_ai/agents/hosting),
which adapts the compiled LangGraph runnable into a REST endpoint
compatible with the Azure AI Invocations protocol. It supports both
streaming (SSE token deltas) and non-streaming (JSON) response modes.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the README in the
parent directory to run the agent host.

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent
> using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please
> refer to the [parent README](../../README.md) for more details. Use
> this README for sample queries you can send to the agent.

Send a POST request with a JSON body containing a `"message"` field.
The `-i` flag includes the response headers, which carry the
`x-agent-session-id` header you need for multi-turn conversations:

```bash
curl -i -X POST http://127.0.0.1:8088/invocations \
  -H "Content-Type: application/json" \
  -d '{"message": "My name is Alice."}'
```

Example response:

```
HTTP/1.1 200
content-type: application/json
x-agent-session-id: 9370b9d4-cd13-4436-a57f-03b843ac0e17
x-platform-server: azure-ai-agentserver-core/... (python/3.12)

{"response": "Nice to meet you, Alice!"}
```

### Multi-turn conversation

Take the `x-agent-session-id` from the previous response and pass it as
a URL parameter on the next request:

```bash
curl -X POST 'http://127.0.0.1:8088/invocations?agent_session_id=9370b9d4-cd13-4436-a57f-03b843ac0e17' \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my name?"}'
```

The agent recalls the previous turn from the `MemorySaver` checkpointer
and answers `"Alice"`.

### Streaming

Add `"stream": true` to receive per-token text deltas as SSE `data:`
lines, followed by `event: done`:

```bash
curl -N -X POST http://127.0.0.1:8088/invocations \
  -H "Content-Type: application/json" \
  -d '{"message": "Count to 5.", "stream": true}'
```

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory.

> The `MemorySaver` checkpointer is in-process only — session state
> will not survive container restarts. For production, swap it for a
> durable checkpointer backend.
