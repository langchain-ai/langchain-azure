# Foundry Toolbox Skills

A LangChain agent that loads skills on demand from an **Azure AI Foundry
Toolbox** and serves the **Responses protocol**. Use it when you want an agent
to discover and follow Foundry-managed skills without loading remote Toolbox
tools. For loading remote Toolbox tools instead, see
[Foundry Toolbox](../04_foundry_toolbox/README.md).

## What This Sample Demonstrates

The included [release-notes skill](skill-src/release-notes/SKILL.md) turns raw
commits or change notes into structured customer-facing release notes. The
agent advertises only the Toolbox skill catalog at startup and reads `SKILL.md`
on demand when a request matches.

## How It Works

At startup, [main.py](main.py) calls
`AzureAIProjectToolbox.aget_resources(uris="skill://index.json")` to read the
Toolbox's skill catalog. The catalog advertises skill names, descriptions, and
resource URIs in the system prompt while keeping skill bodies out of the prompt
until needed. No remote Toolbox tools are fetched (`get_tools()` is not called).

The agent receives one local LangChain `@tool`, `read_skill_resource`:

- When a skill matches the request, the model calls `read_skill_resource` with
  the skill's `SKILL.md` URI. It fetches the content from the Toolbox via
  `aget_resources()` and returns it as a tool message.
- The model follows the loaded procedure directly to produce the answer. It can
  read supporting text resources through the same tool when needed.
- Unrelated requests answer normally without reading a skill.

This follows Foundry's
[progressive-disclosure pattern](https://learn.microsoft.com/azure/foundry/agents/how-to/tools/skills#consume-toolbox-skills-in-microsoft-agent-framework)
using `create_agent` and the SDK's resource API.

All calls target the Toolbox's default version unless you pass
`toolbox_version` to `AzureAIProjectToolbox`. The catalog is loaded once per
process; skill content is fetched on demand. Restart the host after changing
skill metadata or attaching new skills to refresh the catalog. The reader tool
accepts only `skill://` URIs from this Toolbox; it does not read local host
files or execute skill scripts.

The chat model is `langchain_openai.ChatOpenAI`, authenticated with
`DefaultAzureCredential` through the Foundry project endpoint. The compiled
graph is hosted by `ResponsesHostServer`. Model calls set `store=False` because
LangGraph supplies the conversation history on each turn; this setting is
separate from the host's Responses storage.

## Running the Agent Host

### Configure the Skill and Toolbox

Use a Foundry project with a model deployment. The caller needs **Foundry
User** on that project.

From this sample directory, with `FOUNDRY_PROJECT_ENDPOINT` and `TOOLBOX_NAME`
set in your shell (for example `agent-skills`), upload the included
[release-notes skill](skill-src/release-notes/SKILL.md):

```bash
az login
azd extension install microsoft.foundry
azd ai skill create release-notes \
  --file ./skill-src/release-notes \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
```

Create a skills-only Toolbox from [toolbox.yaml](toolbox.yaml) (update `name`
in the file if you use a different `TOOLBOX_NAME`):

```bash
azd ai toolbox create "$TOOLBOX_NAME" \
  --from-file ./toolbox.yaml \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
```

If you already have a Toolbox, attach the skill to a new version based on its
current `default_version` and publish that version:

```bash
azd ai toolbox show "$TOOLBOX_NAME" \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --output json
azd ai toolbox skill add "$TOOLBOX_NAME" release-notes \
  --from-version "<default-version>" \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
azd ai toolbox publish "$TOOLBOX_NAME" "<new-version>" \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
```

To update an existing skill, upload a new version:

```bash
azd ai skill update release-notes --file ./skill-src/release-notes \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
```

An unpinned reference follows the skill's default version. To pin one, attach
`release-notes@<skill-version>` instead. Skills must be in the same project as
the Toolbox. See the Foundry guides for
[skills](https://learn.microsoft.com/azure/foundry/agents/how-to/tools/skills)
and [attaching them to a Toolbox](https://learn.microsoft.com/azure/foundry/agents/how-to/tools/toolbox#attach-skills-to-a-toolbox).
The files in `skill-src/` are upload sources; the host reads skills from the
Toolbox and has no local fallback.

### Run Locally

Follow the parent's [Python setup](../../README.md#using-python) to create a
virtual environment and fill in `.env` from [.env.example](.env.example).
Install this sample's [requirements](requirements.txt) from published packages;
it requires `langchain-azure-ai>=1.3.1` with the `hosting` and `opentelemetry`
extras, plus `langchain-mcp-adapters` and `httpx` for Toolbox MCP access.

Required environment variables:

- `FOUNDRY_PROJECT_ENDPOINT`: your Foundry project endpoint.
- `TOOLBOX_NAME`: the configured Toolbox name (`agent-skills` in `.env.example`).
- `AZURE_AI_MODEL_DEPLOYMENT_NAME`: your model deployment (defaults to `gpt-4o`).

Local runs use the developer's credential; the deployed agent identity also
needs access to the project's model and Toolbox. Follow the shared deployment
guidance for identity setup. Telemetry is enabled by default through the
[shared tracing setup](../../README.md#tracing): spans export to Azure Monitor
(project-managed Application Insights) by default, or to OTLP/HTTP when
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` or `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
Message content, tool arguments, and tool results are redacted unless
`AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true` is set.

From this sample directory, start the host:

```bash
pip install -r requirements.txt
python main.py
```

### Local Checks

From the repository root, run the registered offline smoke checks:

```bash
uv run --project .github/tests --frozen pytest .github/tests --docs-samples-dir samples/hosting/langgraph-hosted-agents/responses/14_foundry_toolbox_skills
```

The tests use the real SDK resource loader, graph, and Responses host with a
scripted model and a fake MCP boundary. They verify metadata-only startup,
on-demand skill reads, error handling, and JSON/SSE responses without Azure
calls.

## Interacting with the Agent

Send a request that matches the `release-notes` skill:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Draft customer release notes for v2.0 from these commits: feat: add --timeout flag (default 30s); fix: retry transient 429 responses; breaking: drop Python 3.10 support, require 3.11+."}'
```

The response should include a `read_skill_resource` call for
`skill://release-notes/SKILL.md`, followed by release notes organized under
**Highlights**, **Fixes**, **Breaking Changes**, and **Upgrade Steps**.

To stream the same flow over SSE:

```bash
curl -N -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Draft customer release notes for v2.0 from these commits: feat: add --timeout flag (default 30s); fix: retry transient 429 responses; breaking: drop Python 3.10 support, require 3.11+.","stream":true}'
```

Skill reads appear as `function_call` / `function_call_output` items over JSON
or SSE. Unrelated requests such as `Say hello in one sentence.` answer directly
without calling `read_skill_resource`.

## Deploying the Agent to Foundry

The [azure.yaml](azure.yaml) uses this sample's [Dockerfile](Dockerfile) and a
remote Azure Container Registry (ACR) build. Reuse an existing Foundry project,
model deployment, and the Toolbox configured above. The project also needs an
ACR connection that the hosted agent can use to pull the image; see the
[Foundry deployment guide](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent)
for registry and identity setup.

With the Foundry extension installed, initialize from an empty directory. Set
`SAMPLE_DIR` to the absolute path of this sample, `FOUNDRY_PROJECT_ID` to your
project's ARM resource ID, and `ACR_CONNECTION_NAME` to its existing registry
connection name:

```bash
azd ai agent init -t "$SAMPLE_DIR/azure.yaml" \
  --project-id "$FOUNDRY_PROJECT_ID" \
  --model-deployment "$AZURE_AI_MODEL_DEPLOYMENT_NAME" \
  --deploy-mode container --acr-connection "$ACR_CONNECTION_NAME" --no-prompt
```

Change to the project directory printed by `init`, then set the Toolbox name in
the azd environment and deploy:

```bash
azd env set TOOLBOX_NAME "$TOOLBOX_NAME"
azd deploy --no-prompt
azd ai agent show --output json
```

No `azd provision` is needed for these existing resources. Initialization binds
the project, model, and registry; `azure.yaml` passes the model deployment and
Toolbox name into the container. The image installs the published SDK from
[requirements.txt](requirements.txt).

To test the deployed agent, send the same requests above to the Responses
endpoint returned by `show`, including its `api-version=v1` query parameter.
Add an `Authorization: Bearer <token>` header with an Entra token for
`https://ai.azure.com`.
