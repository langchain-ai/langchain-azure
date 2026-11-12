# Azure OpenAI Decisions

Ask predicate, choice, and score questions in one request through the Foundry
Azure OpenAI `/openai/v1/decisions` API. This sample uses an experimental,
unreleased integration and requires a supported Azure OpenAI model deployment
(currently `gpt-6-luna`). Microsoft-Decision-1 is outside its scope.

From the repository root, install the checkout containing this change:

```bash
pip install -e ./libs/azure-ai
```

Set these environment variables, using your Azure deployment name:

```text
AZURE_AI_OPENAI_ENDPOINT=https://my-resource.services.ai.azure.com/openai/v1
AZURE_OPENAI_DEPLOYMENT_NAME=my-luna-deployment
```

Sign in with `az login` to use `DefaultAzureCredential`, or set
`AZURE_OPENAI_API_KEY` for key authentication. Then run:

```bash
python samples/decisions/basic_usage.py
```

The script prints each answer's type and values, including refusals, and reported
token usage. See the [integration guide](../../libs/azure-ai/docs/decisions.md)
for project endpoints, async calls, authentication, and Runnable composition.
