# LangChain middleware on Foundry

Host a normal `create_agent` without rewriting its middleware:

- Built-in: [HumanInTheLoopMiddleware](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) with approve/edit/reject.
- Custom: the official [decorator-based message-limit example](https://docs.langchain.com/oss/python/langchain/middleware/custom#node-style-hooks), a single `@before_model` function that stops at 50 messages.

The email tool is simulated; no real email is sent. [agent.py](agent.py) holds
the native agent, [main.py](main.py) owns startup resources and runs the stock
`ResponsesHostServer`, and [client.py](client.py) handles human decisions.

## Run locally

Requires Python 3.11+, a Foundry project, and a tool-capable model deployment.
From this directory, create a virtual environment, then:

```bash
python -m pip install -r requirements.txt
```

Copy [.env.example](.env.example) to `.env`, fill in the project endpoint and
model deployment, authenticate with `az login`, and start the host:

```bash
python main.py
```

The sample pins published `langchain-azure-ai==1.2.10`; no SDK patch is needed.
Direct dependency pins are not a complete transitive lock. Never commit `.env`.

### Tracing

Follow the [shared tracing setup](../../README.md#tracing) for OTLP or
Application Insights. Set `OTEL_SDK_DISABLED=true` to skip tracing setup.
Both independent content-capture switches default to `false`:

| Environment variable | Tracer |
| --- | --- |
| `AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED` | SDK LangChain |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | Agent Server Microsoft LangChain |

Explicit settings are preserved. Enabling either switch can export user
messages, email arguments, and tool results; use only content safe to record.
Startup closes its credentials, transports, checkpoint saver, and any locally
created OTLP provider on exit. An existing platform provider is not replaced.

## Review an action

In another terminal:

```bash
python client.py
```

The client displays the proposed actions, then waits for approve/edit/reject.
No tool executes before approval. Edit changes only the recipient; reject
returns feedback without executing the tool. To exercise each case explicitly:

```bash
python client.py --decision approve
python client.py --decision edit --recipient bob@example.com --stream
python client.py --decision reject
```

A decision applies to all displayed actions in order. The client handles one
review round, not an unattended approval loop; inspect any further review.

## Run on Foundry

Follow the [shared deployment steps](../../README.md#deploying-the-agent-to-foundry)
with this folder's [agent.manifest.yaml](agent.manifest.yaml). After selecting
the intended subscription and project, use container deployment to retain the
Dockerfile's `python main.py` startup:

```bash
azd ai agent init -m ./agent.manifest.yaml --deploy-mode container
azd deploy
```

Grant the deployed identity access to the model and Foundry state storage.
The sample does not create resources or assign roles itself. For code deployment,
set the generated service entry point to `main.py`.

```bash
python client.py --agent-name <deployed-agent-name> --decision approve
python client.py --agent-name <deployed-agent-name> --decision edit --recipient bob@example.com --stream
python client.py --agent-name <deployed-agent-name> --decision reject
```

The client uses the project's Responses endpoint with `agent_reference`,
including on resume. Local checkpoints use `InMemorySaver`; deployed checkpoints
use `FoundryCheckpointSaver`. This sample does not require existing applications
to change their persistence choices.

## Resume transport

The client uses the published SDK's existing rich `function_call_output`
channel, not the MCP boolean shortcut. It reads the paired
`__hosted_agent_adapter_interrupt__` function call and returns an
interrupt-keyed LangGraph resume map containing native decisions:

```json
{
  "type": "function_call_output",
  "call_id": "<interrupt-id>",
  "output": "{\"resume\": {\"<interrupt-id>\": {\"decisions\": [{\"type\": \"approve\"}]}}}"
}
```

The pinned LangChain version uses `action_requests[].args`
([contract](https://github.com/langchain-ai/langchain/blob/f4bc5031dbcf24edb0374a07830915a285222567/libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py)).
A completed response can still contain a pause: inspect output items before
assuming the tool ran. Native rejection is feedback, not the transport error
from the manual-interrupt [08_hitl](../08_hitl/) example.

## Verification

From `libs/azure-ai` with the hosting and opentelemetry extras:

```bash
python -m pytest tests/unit_tests/agents/hosting/hitl -k middleware_sample -q
```

Tests exercise real middleware over JSON/SSE, approval decisions, the 49/50-message
boundary, checkpoint restoration, startup cleanup, and both real tracers' privacy
defaults. Model and storage calls use offline stand-ins. Live Azure, Foundry,
Docker, and client UI end-to-end acceptance are not claimed; run the deployed
client cases before claiming cloud validation.
