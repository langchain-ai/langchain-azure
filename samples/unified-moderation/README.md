# Unified moderation guardrails

Use `AzureContentSafetyPolicyMiddleware` to apply a Content Safety guardrail
policy to a LangChain `create_agent` agent.

## Installation

Use Python 3.11 or later and install the published integration package in
a virtual environment. Unified moderation requires version 1.2.11 or later:

```bash
python -m pip install --upgrade "langchain-azure-ai>=1.2.11"
```

The package installs `azure-ai-contentsafety>=1.1.0b1`. The Content Safety
API remains in preview. Upgrading only that dependency does not add the
middleware to an older `langchain-azure-ai` package.

## Configuration

- Create a Content Safety guardrail policy and set
  `AZURE_CONTENT_SAFETY_POLICY_ID` to its ID.
- Set `AZURE_CONTENT_SAFETY_ENDPOINT` to the Content Safety endpoint.
- Set `AZURE_AI_MODEL` to the model identifier, such as `azure_ai:gpt-4.1`,
  and configure the endpoint and credentials required by that model provider.
- For an `azure_ai` model, set `AZURE_AI_PROJECT_ENDPOINT` to the Foundry
  project endpoint containing the model deployment.
- Sign in with `az login` for the sample's default Azure identity authentication.
  The identity needs access to the model and the Content Safety policy.

## Run

From the repository root:

```bash
python samples/unified-moderation/basic_usage.py
```

The sample prints the agent's response and the Content Safety annotations.
Installation and imports do not validate your Azure resources, permissions,
or live policy behavior.
