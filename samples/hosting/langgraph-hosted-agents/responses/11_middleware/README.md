# What this sample demonstrates

A LangChain `create_agent` with built-in and custom middleware, hosted using
the **Responses protocol**. The email tool is simulated; no real email is sent.

## How It Works

### Model Integration

[main.py](main.py) connects `ChatOpenAI` to the Foundry project's model endpoint
using `DefaultAzureCredential`, `use_responses_api=True`, and
`output_version="responses/v1"`.

### Middleware

[agent.py](agent.py) combines:

- Built-in [HumanInTheLoopMiddleware](https://docs.langchain.com/oss/python/langchain/human-in-the-loop), which pauses email actions for approve/edit/reject.
- One custom [`@before_model` hook](https://docs.langchain.com/oss/python/langchain/middleware/custom#node-style-hooks), which stops the turn at 50 accumulated messages before a model call.

The message check is not a pre-tool limit: a reviewed tool can execute on resume
before the next check stops the following model call.

### Agent Hosting

[main.py](main.py) hosts the graph with `ResponsesHostServer` and
`FoundryCheckpointSaver`: local files during development, Foundry State Store
when deployed. Local checkpoints survive host restarts.
[client.py](client.py) sends approval replies and edited arguments to resume it.

## Running the Agent Host

Use Python 3.11 or later and follow the parent's
[local setup instructions](../../README.md#running-the-agent-host-locally),
installing this sample's [requirements.txt](requirements.txt) and configuring
[.env.example](.env.example). Then run:

```bash
python main.py
```

See the [shared tracing guide](../../README.md#tracing) for exporter setup. Both
`AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED` and
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` default to `false` here.
Enabling either can export messages, tool arguments, and results.

## Interacting with the agent

Run `python client.py` in another terminal to review a proposed email, or choose
a decision explicitly:

```bash
python client.py --decision approve
python client.py --decision edit --recipient bob@example.com
python client.py --decision reject
```

The tool waits for review. Edit changes only the recipient; reject returns
feedback without executing it. A decision applies to all displayed actions.
Edits preserve the original model call and add the executed arguments and a
human-review notice to the tool result.
The client handles one review round and reports if another review is needed.

### Streaming

Add `--stream` to use SSE for both the initial request and resume:

```bash
python client.py --decision edit --recipient bob@example.com --stream
```

### Trigger the custom hook

With the local host running, send 50 messages:

```python
from openai import OpenAI

with OpenAI(base_url="http://127.0.0.1:8088/", api_key="local") as client:
    response = client.responses.create(
        input=[{"role": "user", "content": "Check the message limit."}] * 50,
    )
    print(response.model_dump_json(indent=2))
```

Expect `Conversation limit reached.` without a model or tool call.

## Deploying the Agent to Foundry

Follow the parent's [deployment instructions](../../README.md#deploying-the-agent-to-foundry).
The agent identity needs access to the model and Foundry state storage.

Review a deployed agent with the same client:

```bash
python client.py --agent-name <deployed-agent-name> --decision edit --recipient bob@example.com --stream
```
