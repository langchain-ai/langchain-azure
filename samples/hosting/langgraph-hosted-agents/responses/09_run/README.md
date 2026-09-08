# What this sample demonstrates

A minimal [LangGraph](https://langchain-ai.github.io/langgraph/) agent
built with `langchain.agents.create_agent` and hosted using the
**Responses protocol**. It uses the configuration-driven
`langchain_azure_ai.agents.hosting.run` entrypoint instead of constructing
a `ResponsesHostServer` in application code.

> **No agent code changes are required to migrate an existing LangChain
> agent to this approach.** Keep the existing graph implementation as-is,
> point `langgraph.json` at its exported graph, and launch the hosting
> entrypoint with the desired protocol.

## How It Works

### Model Integration

The agent uses `langchain_openai.ChatOpenAI` with an Azure bearer token
provider from `DefaultAzureCredential` and an OpenAI-compatible endpoint
from `azure.ai.projects.AIProjectClient` (`az login` is enough for local
development). The graph is the same no-tool `create_agent` graph used by
[`responses/01_basic`](../01_basic/).

See [main.py](main.py) for the graph and [langgraph.json](langgraph.json)
for its entrypoint configuration.

### Migrating an Existing Agent

An existing LangChain agent that exports a compiled graph does not need a
hosting wrapper, a `ResponsesHostServer` import, or a new `main()` function.
Add a `langgraph.json` file that references the existing graph symbol, then
use the command shown below. The agent's model, tools, prompts, state, and
graph code remain unchanged.

### Agent Hosting

The `langgraph.json` file maps the `agent` name to the compiled graph in
`main.py`. The hosting entrypoint loads that graph and exposes it through
the protocol selected by the required `--protocol` argument. This sample
selects `responses` in both its local command and Docker image.

## Running the Agent Host

Follow the environment setup instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the
parent README, then start this sample with:

```bash
python -m langchain_azure_ai.agents.hosting.run --protocol responses
```

## Interacting with the agent

Send a POST request with an `"input"` field:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!"}'
```

### Streaming

Add `"stream": true` to receive SSE events as the model produces tokens:

```bash
curl -N -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "stream": true}'
```

### Multi-turn conversation

Include the previous response ID to continue a conversation:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "How are you?", "previous_response_id": "REPLACE_WITH_PREVIOUS_RESPONSE_ID"}'
```

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the [Deploying the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of the
parent README.