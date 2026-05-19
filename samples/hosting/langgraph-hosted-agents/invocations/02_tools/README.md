# What this sample demonstrates

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent with a
**locally-defined Python tool** hosted using the **Invocations
protocol**. Same shape as [`invocations/01_basic`](../01_basic/) but
with a `@tool` function attached to the graph so the agent runs a tool
round-trip before answering.

The Invocations protocol surfaces **only the final assistant text** (or
its token deltas when streaming); the tool call and result pair lives
entirely inside the graph and is consumed during the ReAct loop. If you
need clients to see intermediate tool calls, host the same graph under
the Responses protocol instead (see
[`responses/02_tools`](../../responses/02_tools/)).

## How It Works

### Model Integration

The agent uses `langchain_openai.AzureChatOpenAI` against the Foundry
project endpoint, authenticated with
`DefaultAzureCredential`. The graph is
`create_agent(model, tools=[get_weather], checkpointer=MemorySaver())`,
so every turn runs the standard LangChain ReAct loop with a
`MemorySaver` keeping per-session history.

See [main.py](main.py) for the full implementation.

### Tools

Local tools are plain Python functions decorated with
`langchain_core.tools.tool`. When the model chooses to call a tool, the
agent executes the function server-side and feeds the result back as a
`ToolMessage` on the next iteration of the ReAct loop. Because this
host exposes the Invocations protocol, the call and its result are
**not** surfaced to the client — only the final assistant text (or its
token deltas) is returned.

### Agent Hosting

The agent is hosted using
[`langchain_azure_ai.agents.hosting.LangGraphInvocationsHostServer`](../../../../libs/azure-ai/langchain_azure_ai/agents/hosting),
which adapts the compiled LangGraph runnable into a REST endpoint
compatible with the Azure AI Invocations protocol.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the README in the
parent directory to run the agent host.

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent
> using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please
> refer to the [parent README](../../README.md) for more details. Use
> this README for sample queries you can send to the agent.

Send a POST request with a `"message"` field that triggers the tool:

```bash
curl -X POST http://127.0.0.1:8088/invocations \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Seattle?"}'
```

Example response (the tool ran server-side; the client sees only the
final answer):

```json
{"response": "The weather in Seattle, US is sunny with a high of 22C."}
```

### Streaming

Add `"stream": true` to receive per-token text deltas as SSE `data:`
lines, followed by `event: done`:

```bash
curl -N -X POST http://127.0.0.1:8088/invocations \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Tokyo?", "stream": true}'
```

### Multi-turn conversation

The handler returns an `x-agent-session-id` response header on the
first turn. Reuse it as a URL parameter to continue the conversation
under the same `MemorySaver` thread:

```bash
curl -X POST 'http://127.0.0.1:8088/invocations?agent_session_id=<id>' \
  -H "Content-Type: application/json" \
  -d '{"message": "And in Vancouver?"}'
```

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory.

> The `MemorySaver` checkpointer is in-process only — session state
> will not survive container restarts. For production, swap it for a
> durable checkpointer backend.
