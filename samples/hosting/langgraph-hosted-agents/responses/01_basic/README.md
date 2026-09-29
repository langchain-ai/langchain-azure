# What this sample demonstrates

A minimal [LangGraph](https://langchain-ai.github.io/langgraph/) agent
built with `langchain.agents.create_agent` and hosted using the
**Responses protocol**. No tools, no checkpointer — the smallest possible
host wrapping a LangChain chat model.

## How It Works

### Model Integration

The agent uses `langchain_openai.ChatOpenAI` with an Azure bearer token
provider from `DefaultAzureCredential` and an OpenAI-compatible endpoint
from `azure.ai.projects.AIProjectClient` (`az login`
is enough for local dev). The
underlying graph is a stock `create_agent(model, tools=[])`, so every
turn is just one Responses API call.

See [main.py](main.py) for the full implementation.

### Agent Hosting

The agent is hosted using
[`langchain_azure_ai.agents.hosting.ResponsesHostServer`](../../../../libs/azure-ai/langchain_azure_ai/agents/hosting),
which adapts the compiled LangGraph runnable into a REST endpoint
compatible with the OpenAI Responses protocol. It supports both
streaming (SSE events) and non-streaming (JSON) response modes.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the README in the
parent directory to run the agent host.

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent
> using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please
> refer to the [parent README](../../README.md) for more details. Use
> this README for sample queries you can send to the agent.

Send a POST request to the server with a JSON body containing an
`"input"` field to interact with the agent. For example:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!"}'
```

The server responds with a JSON object containing the assistant message
and a response ID you can reuse to continue the conversation.

### Streaming

Add `"stream": true` to the body to receive SSE events as the model
produces tokens:

```bash
curl -N -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "stream": true}'
```

### Multi-turn conversation

To have a multi-turn conversation, include the previous response id in
the request body:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "How are you?", "previous_response_id": "REPLACE_WITH_PREVIOUS_RESPONSE_ID"}'
```

### Opt-in checkpoint branches

This sample has no checkpointer by default, so its history is message-based.
To restore graph state at an exact completed response, update the graph and
host construction in `main()` to opt in:

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = create_agent(_build_chat_model(), tools=[], checkpointer=InMemorySaver())
port = int(os.environ.get("PORT", "8088"))
StateStoreProbeResponsesHostServer(graph, enable_response_branching=True).run(port=port)
```

Create response A with the first request above. To create two branches from A,
use A's returned ID in both requests, even after B has completed:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Explore option B", "previous_response_id": "REPLACE_WITH_A_ID"}'

curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Explore option C", "previous_response_id": "REPLACE_WITH_A_ID"}'
```

The second branch restores A's graph state, not A followed by B. To regenerate
B, resend B's input with A as the parent; the result has a new response ID.
Referencing B instead continues after B. Do not combine `conversation` with
`previous_response_id`, and resend top-level `instructions` on each request that
needs them. Explicit system/developer input messages remain part of history.

Checkpoint-backed request instructions require a messages reducer that preserves
message IDs and `additional_kwargs`, such as ordinary `add_messages`.
`add_messages(format="langchain-openai")` discards these fields. If a request
instruction's identity is lost, a continuation or task recovery fails before graph
execution instead of silently inheriting the old instructions. Use a reducer that
preserves these fields and start a new response without a parent or conversation;
retrying the same checkpoint cannot restore the lost identity. This restriction
applies with or without `enable_response_branching`. Older checkpoints with
unverifiable instruction provenance are also rejected.

`InMemorySaver` is only suitable for this single-process example. Production
requires persistent graph, response, and branch-record storage. The selected
parent must be stored, completed, and have an available checkpoint; foreground
`store=false` responses cannot become reusable parents. Steering and independent
historical approval forks are not supported. Agent Server SDK `2.1.0b2` also
rejects `background=true, store=false`, despite OpenAI permitting temporary
retention. See the [design and verification notes](../../../../../libs/azure-ai/docs/hosting/time_travel_support.md)
for recovery, ownership retention, and remaining compatibility limits.

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory.
