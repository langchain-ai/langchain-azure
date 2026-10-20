# What this sample demonstrates

The official LangChain [Deep Research example](https://github.com/langchain-ai/deepagents/tree/b6504f6654ee6f25e1cd923087291de9e51572cc/examples/deep_research),
adapted to Microsoft Foundry and hosted over the **Responses protocol**.
It demonstrates planning, delegated web research, and checkpoint-backed working
files without changing the native Deep Agents hosting contract.

## How It Works

### Deep Research

[agent.py](agent.py) uses `create_deep_agent` with `TodoListMiddleware` and
the upstream workflow: plan, save the question, delegate research, consolidate
citations, write `/final_report.md`, and read back the files to check coverage.
[research_agent/prompts.py](research_agent/prompts.py) condenses the upstream
prompts while preserving these stages and delegation guidance. Search and
delegation budgets are prompt guidance, not enforced execution limits.

[research_agent/tools.py](research_agent/tools.py) replaces upstream Tavily
search and arbitrary webpage fetching with the Foundry Toolbox `web_search`
tool. Only that tool is exposed, even if the Toolbox has other tools.
`think_tool` records concise factual progress, not private reasoning. No Tavily
key, local shell, sandbox, or custom hosting middleware is required.

### Model Integration

[main.py](main.py) connects `ChatOpenAI` to the Foundry project with
`DefaultAzureCredential`. It uses `use_responses_api=True` and
`output_version="v1"` for LangChain message history; this setting is independent
of the hosted Responses protocol version.

### Agent Hosting and State

`ResponsesHostServer` hosts the ordinary compiled graph. `StateBackend` keeps
working files in LangGraph state, not on the host filesystem. `InMemorySaver`
retains them locally until the process exits; `FoundryCheckpointSaver` stores
them with the hosted conversation. Continue with `previous_response_id` to read
the saved report. A new conversation does not share these files. This is not
long-term memory, a public file-download API, or in-flight crash recovery.

## Running the Agent Host

Use Python 3.11 or later and follow the parent's
[local setup instructions](../../README.md#running-the-agent-host-locally).
Copy [.env.example](.env.example) to `.env` and configure a Foundry project,
model deployment, and an existing Toolbox exposing `web_search`. The local
developer needs access to the model and Toolbox. Then run from this directory:

```bash
python -m pip install -r requirements.txt
python main.py
```

Both content-recording switches default to `false` before host construction.
See the [shared tracing guide](../../README.md#tracing) for exporter configuration.
Opting in can export prompts, research queries, tool results, and reports.
Search still sends queries to the configured search service, and checkpoints
still store conversation content; telemetry redaction does not disable either.

## Interacting with the agent

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Research how LangGraph checkpoints support multi-turn agents. Use one researcher, cite sources, and save a short report."}'
```

Expect planning, file, and `task` tool calls followed by a cited report.

Read the saved report in the same conversation:

```bash
curl -X POST http://127.0.0.1:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input":"Read /final_report.md and return its contents without researching again.","previous_response_id":"REPLACE_WITH_PREVIOUS_RESPONSE_ID"}'
```

### Streaming

Add `"stream": true` to the request and use `curl -N`. Check the terminal response
status; HTTP 200 or an SSE connection ending is not proof of completion.

## Deploying the Agent to Foundry

[azure.yaml](azure.yaml) declares a model, a web-search-only Toolbox, and a
Python 3.13 hosted agent. Review the resource configuration and costs, then
initialize an environment and deploy from this directory:

```bash
azd env new deep-research --subscription <subscription-id> --location <region>
azd env set AZURE_AI_MODEL_DEPLOYMENT_NAME gpt-4.1-mini
azd up
```

The hosted identity needs model, Toolbox, and state-storage access; developer
permissions alone do not establish runtime access. See the official
[deployment guide](https://learn.microsoft.com/azure/foundry/agents/how-to/deploy-hosted-agent).

Run the shared JSON/SSE checks for research, report continuation, and conversation
isolation against the deployed agent:

```bash
export FOUNDRY_PROJECT_ENDPOINT="$(azd env get-value FOUNDRY_PROJECT_ENDPOINT)"
export DEEP_AGENTS_NAME=langchain-azure-deep-research
python ../../tests/run_samples_e2e.py responses/12_deep_agents
```

For local checks, unset `DEEP_AGENTS_NAME` and set `FOUNDRY_PROJECT_ENDPOINT` and
`TOOLBOX_NAME` in the shell. The runner launches the local host. Both modes make
real model and search calls and can incur charges.

After testing, list compute sessions with `azd ai agent sessions list` and stop
each test session with `azd ai agent sessions stop <session-id>`. Stopping compute
does not delete the agent, model, Toolbox, or saved conversation state.
