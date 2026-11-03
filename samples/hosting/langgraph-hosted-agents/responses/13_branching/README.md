# What this sample demonstrates

A checkpointed `create_agent` graph hosted as the **Responses protocol**
with `enable_response_branching=True`. Clients use `previous_response_id`
to create independent continuations from a completed response, rather
than continuing from the latest checkpoint in the thread.

## How It Works

### Model Integration

The agent uses `langchain_openai.ChatOpenAI` with an Azure bearer token
provider from `DefaultAzureCredential` and an OpenAI-compatible endpoint
from `azure.ai.projects.AIProjectClient`.

See [main.py](main.py) for the full implementation.

### Branching graph

The [client](client.py) creates a shared starting point, forks two versions
of a Seattle day trip, and continues each version independently:

```text
A: Seattle, no car, vegetarian food
|-- B: museums
|   `-- D: summarize the museum version
`-- C: parks and waterfront
    `-- E: summarize the outdoor version
```

Both B and C set `previous_response_id` to A's response ID. D selects B,
even though C was created more recently; E selects C, even though D was
created more recently. Both branches retain A's shared constraints, but
neither inherits the other branch's activity preference.

Every request returns a new response ID. Keep those IDs on the client to
select which version to continue. Sending another request with the same
parent ID creates another sibling; it does not overwrite that parent.

### Responses hosting

Pass a history-preserving checkpointer to the graph and enable branching
on the host:

```python
graph = create_agent(model, tools=[], checkpointer=checkpointer)
await ResponsesHostServer(graph, enable_response_branching=True).run_async(
    port=8088
)
```

The host restores the exact graph checkpoint saved for the selected
parent response. The client supplies only the new input and the parent
response ID, not the full conversation history or LangGraph checkpoint IDs.

Local runs use `MemorySaver`; Foundry-hosted runs use
`FoundryCheckpointSaver`. Restarting the local process loses its graph
checkpoints, so start a new root response after a restart. Hosted runs
keep graph checkpoints in Foundry State Store.

### Requirements and limits

- Branching is opt-in and disabled by default.
- The parent response must be stored, completed, and have a readable saved
  checkpoint. Its graph must also have finished: branching from a paused
  graph or a pending human approval is not supported.
- The checkpointer must preserve history and implement asynchronous
  `aget_tuple`, `aput`, and `aput_writes`. Both savers used here support them.
- Leave `steerable_conversations` disabled and let `ResponsesHostServer`
  create its own host; branching does not support an external `app`.
- Use `previous_response_id` alone for linkage in this sample. Do not send
  `conversation` in the same request.
- This sample demonstrates foreground branching, not crash recovery of
  in-flight work. See the [resilient sample](../10_resilient/README.md) for
  background recovery.

## Running the Agent Host

This sample requires a checkout of this repository, `uv`, and Python 3.13
for the hosted runtime. The published `langchain-azure-ai==1.2.12` does not
include `enable_response_branching`; build the wheel from this checkout
instead. From the repository root:

```bash
cd libs/azure-ai
uv build --wheel --out-dir ../../samples/hosting/langgraph-hosted-agents/responses/13_branching/wheels
cd ../../samples/hosting/langgraph-hosted-agents/responses/13_branching
uv venv --python 3.13
```

Activate `.venv` as described in the [local setup
guide](../../README.md#running-the-agent-host-locally). Copy
[.env.example](.env.example) to `.env` and set `FOUNDRY_PROJECT_ENDPOINT`
and `AZURE_AI_MODEL_DEPLOYMENT_NAME` to your existing Foundry project and
model deployment. Authenticate with Azure CLI before starting the host:

```bash
uv pip install uv -r requirements.txt
python main.py
```

### Run with azd

With the same virtual environment activated, use `azd` with the Foundry
extension. From this sample's directory, create an environment bound to
your existing project and model deployment:

```bash
azd env new branching-demo --subscription <subscription-id> --location <project-region>
azd env set AZURE_AI_PROJECT_ID "<project-resource-id>"
azd env set FOUNDRY_PROJECT_ENDPOINT "https://<resource>.services.ai.azure.com/api/projects/<project>"
azd env set AZURE_AI_MODEL_DEPLOYMENT_NAME gpt-4.1-mini
azd ai agent run --no-client
```

Use the same project endpoint and model deployment in `.env`. `--no-client`
starts only the host so the branching client below can send its five
requests. Add `--port 8090` if port 8088 is already in use.

## Interacting with the agent

### Create and continue two branches

In another terminal, with the same environment activated, run:

```bash
python client.py
```

The client sends five non-streaming requests and prints each response ID
and assistant reply. D should summarize the museum version, while E
should summarize the outdoor version. Exact wording depends on the model.
All requests set `store: true` so their responses remain selectable.

For a different host address or port:

```bash
python client.py --base-url http://127.0.0.1:8090
```

### Raw Responses requests

Create the shared parent A:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Plan a day in Seattle with no car and vegetarian food.","store":true}'
```

Replace `<A_ID>` below with the returned `id`. Both requests select A,
not each other's response IDs:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Focus this version on museums.","previous_response_id":"<A_ID>","store":true}'

curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Focus this version on parks and the waterfront.","previous_response_id":"<A_ID>","store":true}'
```

To continue either branch, use that branch's returned `id` as the next
`previous_response_id`.

### Streaming

Branch selection works the same way with `"stream": true`. For the raw
requests above, add that field to the body and use `curl -N` to read SSE
events, including `response.output_text.delta` and `response.completed`.

## Deploying the Agent to Foundry

Complete the wheel build and azd environment setup above, then deploy:

```bash
azd deploy
azd ai agent show
```

The [azure.yaml](azure.yaml) uses direct-code deployment with Python 3.13
and remote dependency resolution; Docker and a container registry are not
required. The SDK wheel in `wheels/` is included in the deployment package,
while [.agentignore](.agentignore) excludes local credentials, virtual
environments, and azd state. The agent declares Responses protocol `2.0.0`
for deployment routing and Foundry state-store request context. No
`azd provision` is needed when reusing an existing project and model.

Use the deployed agent's OpenAI protocol base URL, without `/responses`
or the query string:

```bash
python client.py --azure --base-url "https://<resource>.services.ai.azure.com/api/projects/<project>/agents/<agent-name>/endpoint/protocols/openai"
```

`--azure` obtains a token with `DefaultAzureCredential` and adds
`api-version=v1`. The client sends the same A/B/C/D/E requests as the local
example, using `FoundryCheckpointSaver` on the hosted agent. See the
[deployment guide](../../README.md#deploying-the-agent-to-foundry) for
Foundry prerequisites.