# Samples — LangGraph hosting (`langchain_azure_ai.agents.hosting`)

| # | File | What it shows |
|---|------|---------------|
| 1 | [sample_01_responses_basic.py](sample_01_responses_basic.py) | Simplest case: host a `create_react_agent` graph as the Responses API. |
| 2 | [sample_02_responses_tools.py](sample_02_responses_tools.py) | Same graph + a `@tool` function. Intermediate tool calls and tool results are surfaced as `function_call` / `function_call_output` output items in both non-streaming and streaming modes. |
| 3 | [sample_03_invocations_basic.py](sample_03_invocations_basic.py) | Host the same graph as the Invocations API, with a `MemorySaver` checkpointer for multi-turn continuity via `agent_session_id`. |
| 4 | [sample_04_invocations_tools.py](sample_04_invocations_tools.py) | Variant of #3 with a local `@tool` function — the agent runs a tool round-trip server-side and returns the final assistant text. Streaming returns per-token text deltas. |
| 5 | [sample_05_workflow_all_in_one.py](sample_05_workflow_all_in_one.py) | All-in-one: a custom multi-node `StateGraph` (plan → tools → synthesize) with two tools, hosted as **both** the Responses API and the Invocations API on the same port via the `app=` parameter. |
| 6 | [sample_06_responses_hitl.py](sample_06_responses_hitl.py) | Human-in-the-loop: the graph uses `langgraph.types.interrupt` to pause for user input. The pause is surfaced as a `function_call(name="__hosted_agent_adapter_interrupt__")` item; the client resumes by posting a matching `function_call_output` with a JSON `{"resume": ...}` payload on the same conversation. |
| 7 | [sample_07_responses_toolbox.py](sample_07_responses_toolbox.py) | Combines this hosting package with `langchain_azure_ai.tools.AzureAIProjectToolbox` — tools are loaded at startup from an Azure AI Foundry Toolbox (a managed multi-MCP gateway) and bound to a `create_react_agent` graph hosted as the Responses API. |

## Setup

From the sample root (`samples/hosting/`):

```bash
# 1. (Recommended) create and activate a virtual environment
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in at minimum
`AZURE_AI_PROJECT_ENDPOINT`. `AZURE_AI_MODEL_DEPLOYMENT_NAME` (defaults
to `gpt-4o`).

Authentication uses default azure auth — `az login` is the simplest setup.

## Running a sample

From the sample root, after the setup above:

```bash
python sample_01_responses_basic.py
```

Open Agent Inspector with the corresponding API and send a message to trigger the agent or use curl command embedded in each sample header.
