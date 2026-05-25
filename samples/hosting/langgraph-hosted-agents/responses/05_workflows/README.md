# What this sample demonstrates

A custom multi-node [LangGraph](https://langchain-ai.github.io/langgraph/)
`StateGraph` (plan → tools → synthesize) with two `@tool` functions,
hosted as the **Responses protocol**. The Responses endpoint surfaces
every intermediate `function_call` / `function_call_output` / `message`.

This is the showcase sample for workflow-style graphs: instead of the
opinionated `create_agent` ReAct loop, you author the graph yourself and
expose it through the Responses protocol.

## How It Works

### Model Integration

The agent uses `langchain_openai.ChatOpenAI` with an Azure bearer token
provider from `DefaultAzureCredential` and an OpenAI-compatible endpoint
from `azure.ai.projects.AIProjectClient`.

See [main.py](main.py) for the full implementation.

### Workflow graph

The graph is a hand-built `StateGraph` with three nodes:

- **plan** — the LLM (bound to two tools, `get_weather` and `add`)
  decides whether tools are needed.
- **tools** — a standard `ToolNode` executes any requested tool calls.
- **synthesize** — the LLM produces the final assistant message.

A `MemorySaver` checkpointer keeps state across turns when the client
reuses the same `conversation.id`.

### Responses hosting

`langchain_azure_ai.agents.hosting.ResponsesHostServer` exposes the
compiled graph through the Responses protocol:

```python
ResponsesHostServer(graph).run(host=..., port=...)
```

Hitting `/responses` returns the full trace as Responses-protocol output
items.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the README in the
parent directory to run the agent host.

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent
> using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please
> refer to the [parent README](../../README.md) for more details. Use
> this README for sample queries you can send to the agent.

### Responses protocol — full tool round-trip with trace

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the weather in Seattle?"}'
```

The `output` array contains the `function_call` / `function_call_output`
pair from the **tools** node and the final assistant `message` from the
**synthesize** node.

### Streaming

The Responses endpoint supports streaming via `"stream": true` in the
body. The stream emits SSE events for every tool round-trip and token
delta.

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory.

The shipped [agent.manifest.yaml](agent.manifest.yaml) declares the
Responses protocol for deployment routing.
