# What this sample demonstrates

A [LangGraph](https://langchain-ai.github.io/langgraph/) **human-in-the-loop**
agent hosted using the **Responses protocol**. The graph uses
`langgraph.types.interrupt` to pause inside an `ask_human` node when the
LLM decides it needs information only the user can provide. The pause
is surfaced to the client as a special `function_call` output item; the
client resumes the run by posting a matching `function_call_output` with
a JSON `{"resume": ...}` payload on the same `conversation.id`.

## How It Works

### Model Integration

The agent uses `langchain_openai.ChatOpenAI` pointed at the Foundry
project's `/openai/v1` endpoint, authenticated with
`DefaultAzureCredential`.

See [main.py](main.py) for the full implementation.

### Interrupt-based HITL

The LLM is bound to two tools:

- `get_weather` — a normal `@tool` function.
- `AskHuman` — a pydantic-schema "tool" whose only purpose is to let
  the LLM declare *"I need a human answer to this question"*.

When the LLM emits an `AskHuman` call, a dedicated `ask_human` node
catches it and raises `langgraph.types.interrupt(...)`. The graph pauses
and the host serializes the interrupt to a Responses output item named
`__hosted_agent_adapter_interrupt__` with its own `call_id`.

State is persisted by an `InMemorySaver` checkpointer keyed by the
`conversation.id`, so the second request continues the paused run from
exactly where it left off.

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

### Step 1 — ask the agent something that requires human input

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "Ask me where I am, then look up the weather there.", "conversation": {"id": "demo-hitl-1"}}'
```

The response `output` array will contain **two** `function_call` items:

- one with `name == "AskHuman"` — the LLM's own tool call. **Do not**
  reply to this one; the graph closes it itself on resume.
- one with `name == "__hosted_agent_adapter_interrupt__"` — the resume
  handle for the LangGraph interrupt. **Copy its `call_id`** for the
  next request.

### Step 2 — resume with the human's answer

Post a `function_call_output` whose `call_id` matches the sentinel
item's `call_id` (not `AskHuman`'s) and whose `output` is a JSON-encoded
resume payload:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": {"id": "demo-hitl-1"},
    "input": [
      {
        "type": "function_call_output",
        "call_id": "<call_id of the __hosted_agent_adapter_interrupt__ item>",
        "output": "{\"resume\": \"Seattle\"}"
      }
    ]
  }'
```

The agent resumes from the `ask_human` node with `"Seattle"` as the
human answer, finishes the weather lookup, and returns a final
assistant `message` item.

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying
the Agent to
Foundry](../../README.md#deploying-the-agent-to-foundry) section of
the README in the parent directory.

> The `InMemorySaver` checkpointer used here is in-process only —
> conversation state will not survive container restarts. For
> production-grade durable HITL, swap in a persistent checkpointer
> backed by Cosmos DB, Redis, or Azure AI Foundry's managed checkpoint
> storage.
