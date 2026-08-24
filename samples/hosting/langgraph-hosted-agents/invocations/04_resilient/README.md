# What this sample demonstrates

A [LangGraph](https://langchain-ai.github.io/langgraph/) trip-planning agent
hosted on Microsoft Foundry over the **Invocations protocol** using
[`langchain_azure_ai.agents.hosting`](https://github.com/langchain-ai/langchain-azure/tree/main/libs/azure-ai/langchain_azure_ai/agents/hosting).
The sample combines Agent Server's resilient invocation lifecycle with durable
LangGraph checkpoints so an interrupted turn can continue after the host
restarts.

> **Work in progress / experimental.** The resilience APIs and recovery
> behavior demonstrated by this sample may change.

It demonstrates:

- background invocations retrieved by a stable invocation ID;
- exact LangGraph checkpoint recovery after a process restart;
- linear multi-turn sessions linked by `previous_invocation_id`;
- durable human approval before a sensitive tool executes;
- foreground streaming plus retrieval and cancellation routes; and
- client recovery from connection failures, interrupted SSE streams, and HTTP
  `5xx` responses.

## How It Works

### Graph shape

The agent is a real-model `StateGraph`. Flight and hotel searches run
automatically, while `book_trip` pauses at a durable LangGraph `interrupt()`
until the client approves or denies the tool call.

```text
START -> agent -> [search tools | approval] -> agent -> END
```

See [main.py](main.py) for the graph, tools, and hosting configuration.

### Recovery model

Recovery depends on two persistent layers: Agent Server stores the durable
invocation and protocol events, while `AsyncSqliteSaver` stores LangGraph
workflow state. Both must remain available after the host restarts.

### Agent hosting

`InvocationsHostServer` exposes `/invocations` and supports foreground
streaming, background execution, retrieval, and cancellation. The host maps
`agent_session_id` to the LangGraph thread and uses `previous_invocation_id` to
continue the latest completed checkpoint in that session.

## Running the Agent Host

### Prerequisites

- Python 3.12 or later
- [`uv`](https://docs.astral.sh/uv/)
- Azure CLI authenticated with `az login`
- A Microsoft Foundry project and model deployment accessible through
  `DefaultAzureCredential`

See the [parent sample guide](../../README.md#running-the-agent-host-locally)
for general Foundry setup options.

### Configure the environment

Create a `.env` file in this directory:

```dotenv
FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-4.1-mini"
```

### Start the host

From this directory, start the host:

```bash
uv sync
uv run python main.py
```

The Invocations endpoint is available at
`http://127.0.0.1:8088/invocations` by default.

## Interacting with the agent

### Use the Textual client

In another terminal, start the Textual CUI:

```bash
cd client
uv sync
uv run python client.py
```

Ask it to book a trip. The CUI displays the proposed `book_trip` arguments
when the graph pauses; choose **Approve** to continue or **Deny** to reject the
tool call.

The CUI generates an `agent_session_id` at startup, reuses it for every turn,
and links turns with `previous_invocation_id`. It creates a stable invocation ID
for every turn and polls the same invocation ID when the connection is
interrupted. The composer remains available immediately after submission. A new
turn is queued locally until its active parent is accepted, then steers it using
the canonical invocation ID as `previous_invocation_id`.

See [client/client.py](client/client.py) for the recovery client
implementation.

Useful client options:

| Option | Purpose |
| --- | --- |
| `--url` | Host base URL or full Invocations endpoint. Defaults to the local host. |
| `--auth` | Acquire an Azure AI bearer token for a deployed agent. |
| `--reconnect-timeout` | Seconds to keep recovering an interrupted turn. Defaults to 120. |

### Test in Agent Inspector

Once the host is running locally, open **Agent Inspector** in VS Code
(Command Palette: **Foundry Toolkit: Open Agent Inspector**) and send:

```text
Find flights and a hotel for a two-night trip to Paris.
```

Use the Textual client for the approval, reconnect, and crash-recovery flows
because it preserves the stable invocation and session IDs required by this
sample.

## Test crash recovery

Start the host with an isolated Agent Server state directory:

```bash
AGENTSERVER_STATE_ROOT="$PWD/.agentserver-demo" uv run python main.py
```

Start the CUI in another terminal:

```bash
cd client
uv run python client.py --reconnect-timeout 300
```

Enter:

```text
Call simulate_crash, recover, and report the result.
```

The tool terminates the host on its first execution. Restart the host with the
same command, from the same directory, before the client timeout expires. The
CUI polls the same invocation and resumes it from the paired LangGraph
checkpoint; do not submit the original request again.

The local LangGraph database is `checkpoints.sqlite` in this directory. The
host reclaims a stale local replay-stream lock only after Agent Server re-enters
the owning invocation in recovered mode, so manual lock deletion is not
required.

## Protocol reference

### Create and retrieve an invocation

Choose the invocation ID before create and reuse it for all recovery requests:

```bash
curl -X POST \
  "http://127.0.0.1:8088/invocations?agent_session_id=trip-demo" \
  -H "Content-Type: application/json" \
  -H "x-agent-invocation-id: <invocation-id>" \
  -d '{
    "message": "Book a two-night trip to Paris",
    "background": true
  }'
```

A background create returns `202`. Poll the invocation by that same ID until
it reaches a terminal status:

```bash
curl "http://127.0.0.1:8088/invocations/<invocation-id>"
```

For foreground SSE output, send `"stream": true` instead of
`"background": true` and consume events through `event: done`. The two modes
cannot be combined. The sample CUI uses foreground streaming and falls back to
retrieval when the stream is interrupted.

### Approve the booking

The first invocation completes with an `mcp_approval_request` in its `output`
array. Use its `id` in the next invocation, keep the same session, and link the
completed turn with `previous_invocation_id`:

```bash
curl -X POST \
  "http://127.0.0.1:8088/invocations?agent_session_id=trip-demo" \
  -H "Content-Type: application/json" \
  -H "x-agent-invocation-id: <next-invocation-id>" \
  -d '{
    "message": [{
      "type": "mcp_approval_response",
      "approval_request_id": "<approval-request-id>",
      "approve": true
    }],
    "previous_invocation_id": "<invocation-id>",
    "background": true
  }'
```

The sample CUI constructs both approval and denial responses automatically.

### Cancel an invocation

```bash
curl -X POST \
  "http://127.0.0.1:8088/invocations/<invocation-id>/cancel"
```

Cancellation stops future work but does not roll back completed checkpoints or
external effects.

## Recovery contract

### Client behavior

| Condition | Required action |
| --- | --- |
| Connection failure, SSE termination without `event: done`, or HTTP `5xx` | Retrieve the same stable invocation ID until it becomes terminal or the reconnect timeout expires. |
| Retrieval returns `404` before create was admitted | Retry create with the same invocation ID. Never generate a replacement ID. |
| Other HTTP `4xx` or an explicit terminal protocol event | Treat the result as final; do not retry it. |
| Starting the next turn | Reuse the `agent_session_id` and send the latest invocation ID as `previous_invocation_id`. |

Each turn needs a stable `x-agent-invocation-id` chosen before create. Sessions
are linear: a new turn continues from the latest completed invocation rather
than forking an older checkpoint.

### Graph and handler behavior

- Compile the graph with a durable checkpointer that survives process
  replacement and is accessible to every recovering host instance.
- Keep durable workflow data in LangGraph state. Process memory, local caches,
  active HTTP requests, `InvocationContext`, and cancellation events are
  transient.
- Make nodes replay-safe. A crash after an external action but before the next
  paired checkpoint can execute that action again.
- Make external side effects idempotent, or deduplicate them with a stable
  operation key. At-least-once execution applies to writes, payments, email,
  queue publication, and other mutating tool calls.
- Keep checkpointed state serializable and compatible across deployments.

Before using this pattern in production, crash-test every node boundary and
both sides of each external side effect. Replace the sample's local SQLite and
file-backed stores with durable stores suitable for the deployment topology.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8088` | HTTP port for the agent host. |
| `AGENTSERVER_STATE_ROOT` | `~/.agentserver` | Local durable task, invocation, and protocol-event state. Reuse it across local restarts. |
| `CHECKPOINT_DB` | `checkpoints.sqlite` locally; `$HOME/checkpoints.sqlite` when hosted | LangGraph checkpoint database. An explicit value takes precedence. |
| `STEERABLE_CONVERSATIONS` | `false` | Enable server-side active-turn steering support. |
| `FOUNDRY_PROJECT_ENDPOINT` | None | Required Foundry project endpoint. |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | None | Required Foundry model deployment name. |

## Deploying the Agent to Foundry

See the [parent deployment guide](../../README.md#deploying-the-agent-to-foundry)
for the common hosted-agent workflow. This directory is an independent `azd`
project. Its [azure.yaml](azure.yaml) defines both regular and steerable
Invocations services.

Install `azd` and authenticate:

```powershell
azd auth login
```

Create a local deployment configuration from the committed template:

```powershell
Copy-Item .env.example .env
```

Replace the placeholders in `.env`. To use an existing Foundry project,
uncomment and set `FOUNDRY_PROJECT_ENDPOINT` and `AZURE_AI_PROJECT_ID`; its
configured model deployment must already exist.

Create and select a new `azd` environment:

```powershell
azd env new resilient
```

If the target `azd` environment already exists, select it instead. This is also
how you switch away from another currently selected environment:

```powershell
azd env list
azd env select <environment-name>
```

Import `.env` into the selected `azd` environment:

```powershell
azd env set --file .\.env
```

To create a new Foundry project and the model declared in `azure.yaml`, provision
them before deploying:

```powershell
azd provision
azd deploy
```

If `.env` targets an existing Foundry project, do not run `azd provision`.
The project and selected model deployment must already exist; deploy the agents
directly:

```powershell
azd deploy
```

Run CUI against a deployed Microsoft Foundry agent with Azure authentication:

```bash
cd client
uv run python client.py --url "https://<account>.services.ai.azure.com/api/projects/<project>/agents/<agent-name>/endpoint/protocols/invocations?api-version=v1" --auth
```
