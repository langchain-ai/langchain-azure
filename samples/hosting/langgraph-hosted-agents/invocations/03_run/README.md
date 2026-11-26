# What this sample demonstrates

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent hosted
using the **Invocations protocol** with session management. It uses the
configuration-driven `langchain_azure_ai.agents.hosting.run` entrypoint
instead of constructing an `InvocationsHostServer` in application code.

> **No agent code changes are required to migrate an existing LangChain
> agent to this approach.** Keep the existing graph implementation as-is,
> point `langgraph.json` at its exported graph, and launch the hosting
> entrypoint with the desired protocol.

Multi-turn continuity is provided by a LangGraph `MemorySaver`
checkpointer: the resolved `agent_session_id` is forwarded to the graph
as `RunnableConfig.configurable.thread_id`, so each session's history is
preserved in process memory.

## How It Works

### Model Integration

The agent uses `langchain_openai.ChatOpenAI` with an Azure bearer token
provider from `DefaultAzureCredential` and an OpenAI-compatible endpoint
from `azure.ai.projects.AIProjectClient`. The graph is the same
`create_agent(model, tools=[], checkpointer=MemorySaver())` graph used by
[`invocations/01_basic`](../01_basic/).

See [main.py](main.py) for the graph and [langgraph.json](langgraph.json)
for its entrypoint configuration.

### Migrating an Existing Agent

An existing LangChain agent that exports a compiled graph does not need a
hosting wrapper, an `InvocationsHostServer` import, or a new `main()`
function. Add a `langgraph.json` file that references the existing graph
symbol, then use the command shown below. The agent's model, tools,
prompts, state, checkpointer, and graph code remain unchanged.

### Agent Hosting

The `langgraph.json` file maps the `agent` name to the compiled graph in
`main.py`. The hosting entrypoint loads that graph and exposes it through
the protocol selected by the required `--protocol` argument. This sample
selects `invocations` in both its local command and Docker image.

## Running the Agent Host

Follow the environment setup instructions in the [Running the Agent Host
Locally](../../README.md#running-the-agent-host-locally) section of the
parent README, then start this sample with:

```bash
python -m langchain_azure_ai.agents.hosting.run --protocol invocations
```

### Run with azd

With the [azd prerequisites](../../README.md#prerequisites) installed,
initialize from this sample's directory, selecting your Foundry project
and model deployment when prompted:

```bash
azd ai agent init -t ./azure.yaml
azd ai agent run --no-client
```

The host serves on `http://127.0.0.1:8088`.
[azure.yaml](azure.yaml) sets `startupCommand` to the hosting-module command
above. Running `python main.py` only exports the graph and exits.

## Interacting with the agent

Send a POST request with a `"message"` field. The response headers include
the `x-agent-session-id` used for multi-turn conversations:

```bash
curl -i -X POST http://127.0.0.1:8088/invocations \
  -H "Content-Type: application/json" \
  -d '{"message": "My name is Alice."}'
```

### Multi-turn conversation

Pass the returned session ID with the next request:

```bash
curl -X POST 'http://127.0.0.1:8088/invocations?agent_session_id=REPLACE_WITH_SESSION_ID' \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my name?"}'
```

### Streaming

Add `"stream": true` to receive per-token text deltas as SSE events:

```bash
curl -N -X POST http://127.0.0.1:8088/invocations \
  -H "Content-Type: application/json" \
  -d '{"message": "Count to 5.", "stream": true}'
```

## Deploying the Agent to Foundry

After initializing with `azure.yaml` above, deploy with `azd deploy`.
Deployment uses the existing [Dockerfile](Dockerfile), whose `CMD` starts
the same hosting module. See the [deployment guide](../../README.md#deploying-the-agent-to-foundry)
for Foundry prerequisites.

> The `MemorySaver` checkpointer is in-process only. Session state does not
> survive container restarts; use a durable checkpointer in production.
