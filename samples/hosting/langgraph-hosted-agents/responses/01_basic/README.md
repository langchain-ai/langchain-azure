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
turn is just one chat-completion call.

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

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory.
