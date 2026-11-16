# Foundry Toolbox tools and skills

A LangChain agent that loads tools and skills from the same **Foundry Toolbox**
and serves the **Responses protocol**. The included `source-brief` skill guides
the agent through a research task: search for primary sources with `web_search`,
check any arithmetic with `code_interpreter`, then write a brief with citations.

## How it works

At startup, [main.py](main.py) uses `AzureAIProjectToolbox.get_tools()` to load
LangChain tools and `aget_skills(backend=...)` to download skill files into a
temporary `FilesystemBackend`. Both use the Toolbox's default version. Set
`toolbox_version` on `AzureAIProjectToolbox` to pin a specific version.

The existing `create_agent` graph uses two Deep Agents middleware components:

- `SkillsMiddleware` advertises skill names and descriptions in the system
  prompt. Full instructions enter the conversation when the agent reads a skill.
- `FilesystemMiddleware`, restricted to `read_file`, lets the agent read the
  downloaded skills and their references. Paths stay inside the temporary
  directory. Conversation and tool-output offloading are disabled so this shared
  directory contains only Toolbox skills.

For a research request, the expected sequence is **read the skill → search →
calculate if needed → write the brief**. The procedure lives in the skill;
the agent code discovers it and provides the tools to carry it out. A numerical
comparison uses `code_interpreter`; a brief with no arithmetic only needs search.

Skill content is downloaded from Foundry once per host process. Restart the host
after publishing a Toolbox or skill update. If no skills are attached, startup
logs show `Loaded 0 skill file(s)` and ordinary Toolbox tool calls still work.

The model remains `langchain_openai.ChatOpenAI`, authenticated with
`DefaultAzureCredential` through the Foundry project endpoint. The compiled graph
is hosted by `ResponsesHostServer`.

## Configure the Toolbox

Use a Foundry project with a model deployment and a Toolbox containing
`web_search` and `code_interpreter`. The caller needs **Foundry User** on that
project. The [agent manifest](agent.manifest.yaml) declares these two tools;
skill creation and attachment are separate steps below.

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

Follow the parent's [local setup](../../README.md#running-the-agent-host-locally),
then install this sample's [requirements](requirements.txt) and run `python main.py`.
This sample requires `langchain-azure-ai>=1.3.1` and
`deepagents==0.7.28`. If the SDK version is not available on your package index,
install the repository SDK before the sample requirements (from the repo root):

```bash
pip install "./libs/azure-ai[hosting,opentelemetry,tools]"
pip install -r samples/hosting/langgraph-hosted-agents/responses/04_foundry_toolbox/requirements.txt
```

Required environment variables:

- `FOUNDRY_PROJECT_ENDPOINT`: your Foundry project endpoint.
- `TOOLBOX_NAME`: the configured Toolbox name (`agent-tools` in the manifest).
- `AZURE_AI_MODEL_DEPLOYMENT_NAME`: your model deployment (defaults to `gpt-4.1-mini`,
  matching the manifest).

Local runs use the developer's credential; the deployed agent identity also needs
access to the project's model and Toolbox. Follow the shared deployment guidance
for identity setup. Telemetry is enabled by default through the
[shared tracing setup](../../README.md#tracing); message content, tool arguments,
and tool results are redacted unless content recording is explicitly enabled.

Startup lists the loaded tools and skill paths, including
`/skills/source-brief/SKILL.md`.

## Try the skill and tools together

Send a research request:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Use the source-brief skill to compare Python 3.13 and 3.14 for a small web service. Find primary sources and include performance figures only when the measurements are comparable. Use code_interpreter for any percentage calculations."}'
```

The response should include a `read_file` call for the skill, `web_search` calls,
and a brief with **Recommendation**, **Evidence**, and **Open questions**. A
`code_interpreter` call appears when the task needs a calculation.

To exercise both remote tools with explicit calculation inputs, stream a request:

```bash
curl -N -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Use the source-brief skill to research Python 3.14 upgrade considerations for a web service. Our own comparable benchmark measured 250 ms on 3.13 and 200 ms on 3.14. Use web_search for release guidance and code_interpreter to calculate our latency reduction. Keep our measurements separate from published claims.","stream":true}'
```

Here `code_interpreter` should calculate a **20% latency reduction** from the
user's inputs. Intermediate reads and remote tool calls appear as
`function_call` / `function_call_output` items over JSON or SSE. Ordinary tool
requests such as `Use code_interpreter to calculate 17 * 25` still work.

## Deploy to Foundry

Follow the parent's [deployment guide](../../README.md#deploying-the-agent-to-foundry).
After `azd provision` creates the Toolbox, upload and attach the skill using the
steps above, then deploy the agent. `TOOLBOX_NAME` is injected from the manifest.
The image build needs access to a package index containing the required SDK
version. To use an existing Toolbox instead, configure its name for the agent.

## Local checks

From the repository root, run the registered offline smoke checks:

```bash
uv run --project .github/tests --frozen pytest .github/tests --docs-samples-dir samples/hosting/langgraph-hosted-agents/responses/04_foundry_toolbox
```

The tests use the real SDK resource loader, middleware, graph, and Responses
host with a scripted model and a fake MCP boundary. They require no Azure calls.
