# What this sample demonstrates

A custom multi-node [LangGraph](https://langchain-ai.github.io/langgraph/)
`StateGraph` (plan → tools → synthesize) with two `@tool` functions,
hosted as **both** the Responses protocol **and** the Invocations
protocol on the same port. The Responses endpoint surfaces every
intermediate `function_call` / `function_call_output` / `message`; the
Invocations endpoint returns just the final assistant text (or streams
its tokens).

This is the showcase sample for workflow-style graphs: instead of the
opinionated `create_agent` ReAct loop, you author the graph yourself and
choose which protocol(s) to expose it under.

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
reuses the same `agent_session_id` (Invocations) or `conversation.id`
(Responses).

### Multi-protocol hosting

`langchain_azure_ai.agents.hosting` exposes the same compiled graph
under both protocols via a `MultiProtocolHost`:

```python
MultiProtocolHost(
    InvocationAgentServerHost(...),
    ResponsesAgentServerHost(...),
).run(host=..., port=...)
```

Hitting `/responses` returns the full trace as Responses-protocol output
items; hitting `/invocations` returns the same graph's final answer in
Invocations-protocol shape.

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

### Invocations protocol — same graph, final-text-only shape

```bash
curl -i -X POST http://127.0.0.1:8088/invocations \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 17 plus 25?"}'
```

The response is a single JSON object `{"response": "..."}`. The tool
call still happens inside the graph — it's just not surfaced under this
protocol.

### Streaming

Both endpoints support streaming via `"stream": true` in the body. The
Responses stream emits SSE events for every tool round-trip and token
delta; the Invocations stream emits per-token text deltas followed by
`event: done`.

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory.

> The shipped [agent.manifest.yaml](agent.manifest.yaml) declares only
> the Responses protocol for deployment routing because Foundry's
> hosted-agent infrastructure expects a single primary protocol per
> agent. The Invocations endpoint is still served by the running
> container on the same port — it just isn't advertised through the
> hosted-agent control plane.
