# Foundry Toolbox Tools and Skills

A LangChain agent that loads both **tools** and **skills** from an **Azure AI
Foundry Toolbox** and serves the **Responses protocol**. Use it when you want
one Foundry Toolbox to supply remote tools and reusable skills independently.

## What This Sample Demonstrates

This sample demonstrates two independent Toolbox capabilities in one agent:

- **Load tools**: call remote Toolbox tools (such as `web_search` or
  `code_interpreter`) loaded via `AzureAIProjectToolbox.get_tools()`.
- **Load skills**: discover the Toolbox's skill catalog at startup and read an
  attached skill (the included [release-notes skill](skill-src/release-notes/SKILL.md))
  on demand via `AzureAIProjectToolbox.aget_resources()`, without requiring the
  skill to call remote tools.

## How It Works

At startup, [main.py](main.py) connects to the configured Foundry Toolbox with
[`AzureAIProjectToolbox`](../../../../../libs/azure-ai/langchain_azure_ai/tools):

- `await toolbox.get_tools()` fetches the Toolbox's remote tools as LangChain
  `BaseTool` instances.
- `await toolbox.aget_resources(uris="skill://index.json")` fetches the
  Toolbox's skill catalog and places only the metadata (skill names,
  descriptions, and `skill://` URIs) in the system prompt.

The agent also receives a local LangChain `@tool`, `read_skill_resource`:

- For a **tool request**, the model calls the remote Toolbox tools directly
  without reading a skill.
- For a **skill request**, the model calls `read_skill_resource` with the
  matching `SKILL.md` URI. It reads the content from the Toolbox via
  `aget_resources()` and returns it as a tool message, then follows the
  procedure directly. Supporting text files can be read through the same tool
  when needed.

This follows Foundry's
[progressive-disclosure pattern](https://learn.microsoft.com/azure/foundry/agents/how-to/tools/skills#consume-toolbox-skills-in-microsoft-agent-framework)
using the standard LangChain `create_agent` graph and SDK resource API.

All calls target the Toolbox's default version unless you pass
`toolbox_version` to `AzureAIProjectToolbox`. Tools and the skill catalog are
loaded once per process; skill content is fetched on demand. Restart the host
after Toolbox or skill metadata updates to refresh them. Attach `release-notes`
to the Toolbox before starting the host. `read_skill_resource` accepts only
`skill://` URIs from this Toolbox; it does not read local host files or execute
skill scripts.

The chat model is `langchain_openai.ChatOpenAI`, authenticated with
`DefaultAzureCredential` through the Foundry project endpoint. The compiled
graph is hosted by `ResponsesHostServer`. Model calls set `store=False` because
LangGraph supplies the conversation history on each turn; this setting is
separate from the host's Responses storage.

## Running the Agent Host

### Configure the Toolbox

Use a Foundry project with a model deployment and a Toolbox (for example, one
configured with `web_search` and `code_interpreter`). The caller needs
**Foundry User** on that project.

From this sample directory, with `FOUNDRY_PROJECT_ENDPOINT` and `TOOLBOX_NAME`
set in your shell, upload the included
[release-notes skill](skill-src/release-notes/SKILL.md):

```bash
az login
azd extension install microsoft.foundry
azd ai skill create release-notes \
  --file ./skill-src/release-notes \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
azd ai toolbox show "$TOOLBOX_NAME" \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --output json
```

Attach the skill to a new Toolbox version based on its current
`default_version` (replace `<default-version>` with the value from `show`):

```bash
azd ai toolbox skill add "$TOOLBOX_NAME" release-notes \
  --from-version "<default-version>" \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
```

This keeps that version's existing tools and creates a new Toolbox version.
Publish the version returned by `skill add` so the sample can discover it:

```bash
azd ai toolbox publish "$TOOLBOX_NAME" "<new-version>" \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
```

If the skill already exists, upload a new skill version:

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
- `TOOLBOX_NAME`: the configured Toolbox name (`agent-tools` in `.env.example`).
- `AZURE_AI_MODEL_DEPLOYMENT_NAME`: your model deployment (defaults to `gpt-4o`).

Local runs use the developer's credential; the deployed agent identity also
needs access to the project's model and Toolbox. Follow the shared deployment
guidance for identity setup. Telemetry is enabled by default through the
[shared tracing setup](../../README.md#tracing): spans export to Azure Monitor
(project-managed Application Insights) by default, or to OTLP/HTTP when
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` or `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
Message content, tool arguments, and tool results are redacted unless
`AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true` is set.

Startup lists the loaded tools and confirms that the skill catalog was loaded.

From this sample directory, start the host:

```bash
pip install -r requirements.txt
python main.py
```

### Local Checks

From the repository root, run the registered offline smoke checks:

```bash
uv run --project .github/tests --frozen pytest .github/tests --docs-samples-dir samples/hosting/langgraph-hosted-agents/responses/04_foundry_toolbox
```

The tests use the real SDK tool/resource loader, graph, and Responses host with
a scripted model and a fake MCP boundary. They verify metadata-only startup,
direct Toolbox tool calls, on-demand skill reads, and error handling without
Azure calls.

## Interacting with the Agent

### Example 1: Load and Call a Toolbox Tool

Ask the agent to use a remote tool from the Toolbox directly (no skill read
needed):

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Use code_interpreter to calculate 17 * 25."}'
```

The response surfaces `function_call` / `function_call_output` items for
`code_interpreter` and returns **425**. You can also list available tools or
stream any request with `"stream": true`.

### Example 2: Load and Follow a Toolbox Skill

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

Skill reads and remote tool calls appear as `function_call` /
`function_call_output` items over JSON or SSE.

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
[requirements.txt](requirements.txt). The legacy `agent.manifest.yaml` is not
used by this setup; current `azd` requires an `azure.yaml` project.

To test the deployed agent, send the same requests above to the Responses
endpoint returned by `show`, including its `api-version=v1` query parameter.
Add an `Authorization: Bearer <token>` header with an Entra token for
`https://ai.azure.com`.
