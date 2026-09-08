# Responses resilience E2E

This package tests crash recovery in
`langchain_azure_ai.agents.hosting.ResponsesHostServer`. The server uses a
fixed LangGraph workflow and an in-process deterministic chat model that
streams one word at a time. It does not call any external service.

## Mechanism

The workflow is:

```text
START -> 1plan -> 2research -> 3execute -> 4summarize -> END
```

Select a crash point by sending a JSON object as the user message. This works
from any chat client connected to the deployed Foundry agent:

```json
{"crash":"after_2research_checkpoint_before_metadata"}
```

Malformed JSON, a missing `crash`, or an unknown value returns an
instruction with the expected format and allowed values. One deployed server
can therefore run every crash case manually without request-metadata support or
redeployment.

To add another crash case:

1. Add a string constant to `server/server_app/crash_points.py`.
2. Add its event or dependency match in `server/server_app/crash_injection.py`.
3. Add its shared definition under `common/case_definitions/`.

## Run locally


```powershell
uv run --frozen --all-extras --group test pytest -q tests/e2e_tests/agents/hosting/responses_resilience/local/run_case.py -vv
```

## Deploy to Foundry

The project uses the standard direct-code `azd` flow. It provisions a new
Foundry project but does not provision a model.

Configure an `azd` environment once from this directory:

```powershell
azd env new ai-test-e2e `
  --subscription "<subscription-id>" `
  --location "<project-region>" `
  --no-prompt

azd provision --no-prompt
```

Deploy with the normal command:

```powershell
azd deploy langchain-azure-responses-resilience-e2e --no-prompt
```

The `prepackage` hook in `azure.yaml` automatically stages the repository's
current `langchain_azure_ai` source under `server/vendor/`. The remote build
installs that stable source path, so changing the package version does not
require updating a wheel filename or this test project.

## Trigger on Foundry

From this directory after deployment:

```powershell
$endpoint = azd env get-value AGENT_LANGCHAIN_AZURE_RESPONSES_RESILIENCE_E2E_RESPONSES_ENDPOINT `
  -e ai-test-e2e --no-prompt

uv run --frozen --all-extras --group test pytest -q .\remote\run_case.py -vv `
  --responses-endpoint $endpoint `
  --reconnect-timeout 120
```

To run one crash case and capture its full result:

```powershell
uv run --frozen --all-extras python .\remote\run_case.py `
  --url $endpoint `
  --crash-point after_1plan_responses_checkpoint `
  --reconnect-timeout 60 `
  --result-file .\result.json
```

The sample client chooses the response ID before create, tolerates the
connection gap while Foundry replaces the crashed process, and retrieves from
its latest SSE sequence-number cursor until terminal. Success exits with code
`0` and prints/writes:

```json
{
  "recovery_started_seconds": 46.75,
  "requested_response_id": "caresp_...",
  "response_id": "caresp_...",
  "response_status": "completed",
  "result": {
    "node_runs": {
      "1plan": 1,
      "2research": 2,
      "3execute": 1,
      "4summarize": 1
    },
    "checkpoint_writes": {
      "1plan": 1,
      "2research": 1,
      "3execute": 1,
      "4summarize": 1
    }
  }
}
```

A non-terminal response, malformed output, or unexpected run/write count exits
nonzero. Use `azd ai agent monitor --follow` with the same environment to
inspect startup or recovery failures.