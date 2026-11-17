# Foundry Toolbox tools and skills

A LangChain agent that loads tools and skills from the same **Foundry Toolbox**
and serves the **Responses protocol**. The included `source-brief` skill guides
the agent through a research task: search for primary sources with `web_search`,
check any arithmetic with `code_interpreter`, then write a brief with citations.

## How it works

At startup, [main.py](main.py) uses `AzureAIProjectToolbox.get_tools()` to load
LangChain tools and `aget_resources(uris="skill://index.json")` to read the
Toolbox's skill catalog. The catalog advertises skill names, descriptions, and
resource URIs in the system prompt. Skill instructions stay out of the prompt
until the model requests them.

The agent also receives a small LangChain `@tool`, `read_skill_resource`:

- When a skill matches the request, the model calls this tool with the skill's
  `SKILL.md` URI. It reads the content from the Toolbox through `aget_resources()`
  and returns it as a tool message.
- The model follows those instructions using the Toolbox tools. It can read
  supporting text files through the same resource tool when needed.

For `source-brief`, the expected sequence is **read the skill → search →
calculate if needed → write the brief**. A numerical comparison uses
`code_interpreter`; a brief with no arithmetic only needs search. Unrelated
requests can use tools without loading a skill. This follows Foundry's
[progressive-disclosure pattern](https://learn.microsoft.com/azure/foundry/agents/how-to/tools/skills#consume-toolbox-skills-in-microsoft-agent-framework)
using the existing LangChain agent and SDK resource API.

This sample uses the Toolbox's `skill://index.json` discovery resource. All
calls use the Toolbox's default version unless you set `toolbox_version` on
`AzureAIProjectToolbox`. Tools and the catalog are loaded once per process;
skill content is fetched on demand. Restart after Toolbox or skill metadata
updates to refresh the catalog. Attach `source-brief` before starting the host.
The resource tool reads only `skill://` resources from this Toolbox; it does
not read host files or execute skill scripts.

The model remains `langchain_openai.ChatOpenAI`, authenticated with
`DefaultAzureCredential` through the Foundry project endpoint. The compiled graph
is hosted by `ResponsesHostServer`. Model calls set `store=False` because
LangGraph sends the message history, including previous tool calls, on each turn.
This setting is separate from the host's Responses storage.

## Configure the Toolbox

Use a Foundry project with a model deployment and a Toolbox containing
`web_search` and `code_interpreter`. The caller needs **Foundry User** on that
project. Skill creation and attachment are separate steps below.

From this sample directory, with `FOUNDRY_PROJECT_ENDPOINT` and `TOOLBOX_NAME`
set in your shell, upload the included [source-brief skill](skill-src/source-brief/SKILL.md):

```bash
az login
azd extension install microsoft.foundry
azd ai skill create source-brief \
  --file ./skill-src/source-brief \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
azd ai toolbox show "$TOOLBOX_NAME" \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --output json
```

Attach it to a new version based on your Toolbox's current `default_version`.
Replace `<default-version>` with that value from `show`:

```bash
azd ai toolbox skill add "$TOOLBOX_NAME" source-brief \
  --from-version "<default-version>" \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
```

This retains that version's tools and creates a new Toolbox version. Publish
the version returned by `skill add` so the sample can discover it:

```bash
azd ai toolbox publish "$TOOLBOX_NAME" "<new-version>" \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
```

If the skill already exists, create a new skill version:

```bash
azd ai skill update source-brief --file ./skill-src/source-brief \
  --project-endpoint "$FOUNDRY_PROJECT_ENDPOINT" --no-prompt
```

An unpinned reference follows the skill's default version. To pin one, attach
`source-brief@<skill-version>` instead. Skills must be in the same project as
the Toolbox. See the Foundry guides for
[skills](https://learn.microsoft.com/azure/foundry/agents/how-to/tools/skills)
and [attaching them to a Toolbox](https://learn.microsoft.com/azure/foundry/agents/how-to/tools/toolbox#attach-skills-to-a-toolbox).
The files in `skill-src/` are upload sources; the host reads skills from the
Toolbox and has no local fallback.

## Run locally

Follow the parent's [Python setup](../../README.md#using-python),
then install this sample's [requirements](requirements.txt) and run `python main.py`.
This sample requires `langchain-azure-ai>=1.3.1`.
If the SDK version is not available on your package index,
install the repository SDK before the sample requirements (from the repo root):

```bash
pip install "./libs/azure-ai[hosting,opentelemetry,tools]"
pip install -r samples/hosting/langgraph-hosted-agents/responses/04_foundry_toolbox/requirements.txt
```

Required environment variables:

- `FOUNDRY_PROJECT_ENDPOINT`: your Foundry project endpoint.
- `TOOLBOX_NAME`: the configured Toolbox name (`agent-tools` in the manifest).
- `AZURE_AI_MODEL_DEPLOYMENT_NAME`: your model deployment (defaults to `gpt-4o`).

Local runs use the developer's credential; the deployed agent identity also needs
access to the project's model and Toolbox. Follow the shared deployment guidance
for identity setup. Telemetry is enabled by default through the
[shared tracing setup](../../README.md#tracing); message content, tool arguments,
and tool results are redacted unless content recording is explicitly enabled.

Startup lists the loaded tools and confirms that the skill catalog was loaded.

## Try the skill and tools together

Send a research request:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Use the source-brief skill to compare Python 3.13 and 3.14 for a small web service. Find primary sources and include performance figures only when the measurements are comparable. Use code_interpreter for any percentage calculations."}'
```

The response should include a `read_skill_resource` call for
`skill://source-brief/SKILL.md`, `web_search` calls, and a brief with
**Recommendation**, **Evidence**, and **Open questions**. A
`code_interpreter` call appears when the task needs a calculation.

To exercise both remote tools with explicit calculation inputs, stream a request:

```bash
curl -N -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Use the source-brief skill to research Python 3.14 upgrade considerations for a web service. Our own comparable benchmark measured 250 ms on 3.13 and 200 ms on 3.14. Use web_search for release guidance and code_interpreter to calculate our latency reduction. Keep our measurements separate from published claims.","stream":true}'
```

Here `code_interpreter` should calculate a **20% latency reduction** from the
user's inputs. Skill reads and remote tool calls appear as
`function_call` / `function_call_output` items over JSON or SSE. Ordinary tool
requests such as `Use code_interpreter to calculate 17 * 25` still work.

## Deploy to Foundry

The [azure.yaml](azure.yaml) uses the sample's [Dockerfile](Dockerfile) and a
remote Azure Container Registry (ACR) build. Reuse an existing Foundry project,
model deployment, and the Toolbox configured above. The project also needs an
ACR connection that the hosted agent can use to pull the image; see the
[Foundry deployment guide](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent)
for registry and identity setup.

With the Foundry extension installed, initialize from an empty directory.
Set `SAMPLE_DIR` to the absolute path of this sample, `FOUNDRY_PROJECT_ID` to
your project's ARM resource ID, and `ACR_CONNECTION_NAME` to its existing
registry connection name:

```bash
azd ai agent init -t "$SAMPLE_DIR/azure.yaml" \
  --project-id "$FOUNDRY_PROJECT_ID" \
  --model-deployment "$AZURE_AI_MODEL_DEPLOYMENT_NAME" \
  --deploy-mode container --acr-connection "$ACR_CONNECTION_NAME" --no-prompt
```

Change to the project directory printed by `init`, then set the Toolbox name
in the azd environment and deploy:

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

## Local checks

From the repository root, run the registered offline smoke checks:

```bash
uv run --project .github/tests --frozen pytest .github/tests --docs-samples-dir samples/hosting/langgraph-hosted-agents/responses/04_foundry_toolbox
```

The tests use the real SDK resource loader, graph, and Responses
host with a scripted model and a fake MCP boundary. They verify metadata-only
startup, on-demand reads, and tool calls without Azure calls.
