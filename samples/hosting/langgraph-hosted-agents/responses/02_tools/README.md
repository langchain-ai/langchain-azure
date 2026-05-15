# What this sample demonstrates

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent with a
**locally-defined Python tool** hosted using the **Responses protocol**.
It shows how to register a `@tool`-decorated function with
`create_agent` so the model can call it during a conversation, and how
the Responses host surfaces every tool round-trip to the client as
`function_call` / `function_call_output` output items.

## How It Works

### Model Integration

The agent uses `langchain_openai.ChatOpenAI` pointed at the Foundry
project's `/openai/v1` endpoint, authenticated with
`DefaultAzureCredential`. The graph is built by
`langchain.agents.create_agent(model, tools=[get_weather])`, which gives
the LLM access to one local tool.

See [main.py](main.py) for the full implementation.

### Tools

Local tools are plain Python functions decorated with
`langchain_core.tools.tool`. When the model chooses to call a tool, the
agent executes the function and feeds the result back as a
`ToolMessage` on the next iteration of the ReAct loop. The host then
emits both the call and its result to the client as separate output
items:

- `function_call` — the tool name and its arguments.
- `function_call_output` — the tool's return value.

Both items appear in non-streaming JSON responses (in the `output`
array) and in streaming SSE responses (as
`response.output_item.added` / `response.output_item.done` events) so
the client sees the full tool-use trace.

### Agent Hosting

The agent is hosted using
[`langchain_azure_ai.agents.hosting.LangGraphResponsesHostServer`](../../../../libs/azure-ai/langchain_azure_ai/agents/hosting),
which adapts the compiled LangGraph runnable into a REST endpoint
compatible with the OpenAI Responses protocol.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the README in the
parent directory to run the agent host.

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent
> using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please
> refer to the [parent README](../../README.md) for more details. Use
> this README for sample queries you can send to the agent.

Send a POST request with an `"input"` field that triggers the tool:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the weather in Seattle?"}'
```

The JSON `output` array will contain three items in order:

1. `function_call` — `get_weather` with the arguments the LLM chose.
2. `function_call_output` — the string returned by the Python tool.
3. `message` — the final assistant text.

### Streaming

Add `"stream": true` to receive the events as they happen:

```bash
curl -N -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the weather in Tokyo?", "stream": true}'
```

You should see the events arrive in this order:

```
response.output_item.added/done   (function_call)
response.output_item.added/done   (function_call_output)
response.output_item.added        (message)
response.output_text.delta * N
response.output_text.done
response.output_item.done         (message)
response.completed
```

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory.
