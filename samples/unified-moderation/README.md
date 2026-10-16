# Unified moderation guardrails

Use `AzureContentSafetyPolicyMiddleware` to apply a Content Safety guardrail
policy to a LangChain `create_agent` agent.

## Installation

This preview is not included in the published `langchain-azure-ai==1.2.10`
package. Use Python 3.11 or later and Git, and install the pinned source
revision in a virtual environment:

```bash
python -m pip install --upgrade --force-reinstall "langchain-azure-ai @ git+https://github.com/langchain-ai/langchain-azure.git@cb9db867b4e4f7dab74995254efe3332a6b6e2f5#subdirectory=libs/azure-ai"
```

The source package requires `azure-ai-contentsafety>=1.1.0b1`. Upgrading only
that dependency does not add the middleware to an older
`langchain-azure-ai` package. The source pin does not lock every dependency.
Use `--force-reinstall` because this source snapshot and the published package
both use version 1.2.10.

## Configuration

- Create a Content Safety guardrail policy and set
  `AZURE_CONTENT_SAFETY_POLICY_ID` to its ID.
- Set `AZURE_CONTENT_SAFETY_ENDPOINT` to the Content Safety endpoint.
- Set `AZURE_AI_MODEL` to the model identifier, such as `azure_ai:gpt-4.1`,
  and configure the endpoint and credentials required by that model provider.
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
