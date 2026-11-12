# Azure OpenAI Decisions

`AzureAIOpenAIApiDecisions` asks typed questions about shared input using the
[Azure OpenAI Decisions API in Foundry](https://learn.microsoft.com/azure/foundry/openai/how-to/decisions).
It extends `langchain_openai.decisions.OpenAIDecisions` with Azure endpoint
resolution and authentication. Question types, responses, and Runnable methods
come from that upstream integration.

This integration is **experimental** and emits `ExperimentalWarning` when
constructed. Its interface may change. No runtime opt-in flag is required.
The upstream `OpenAIDecisions` interface is beta.

The endpoint is `/openai/v1/decisions`. Use the deployment name of a supported
Azure OpenAI Decisions model, currently `gpt-6-luna`. Microsoft-Decision-1 and
the `/providers/microsoft/v1/systemone` API are outside this integration's scope.

## Install

This API is unreleased. From a checkout containing this change:

```bash
pip install -e ./libs/azure-ai
```

The package requires `langchain-openai>=1.7.2,<2.0.0`. No extra is needed.

## Ask typed questions

Use Microsoft Entra ID with an identity authorized to call the model:

```python
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.decisions import (
    AzureAIOpenAIApiDecisions,
    Choice,
    Predicate,
    Score,
)

decisions = AzureAIOpenAIApiDecisions(
    endpoint="https://my-resource.services.ai.azure.com/openai/v1",
    credential=DefaultAzureCredential(),
    model="my-luna-deployment",
)
request = {
    "input": "Checkout is broken. Please refund my payment.",
    "questions": {
        "refund": Predicate(instructions="Is a refund requested?"),
        "team": Choice(
            instructions="Which team should handle this request?",
            choices={"billing": "Payment issues", "engineering": "Product bugs"},
        ),
        "severity": Score(
            instructions="How severe is the problem?", levels=["Minor", "Major"]
        ),
    },
}
response = decisions.invoke(request)
for name, answer in response.answers.items():
    print(name, answer.model_dump())
```

Answers are keyed by question name. `response.predicates`, `response.choices`,
and `response.scores` contain only their respective answer types.
Check `response.refusals` before accessing a question through a typed mapping:
a refusal is returned there instead. `response.usage` contains reported token
counts. HTTP errors propagate as OpenAI SDK exceptions.

## Configure Azure

Pass either `endpoint` (including `/openai/v1`) or `project_endpoint`.
The inherited `base_url` is accepted instead of `endpoint`.
For a project URL such as
`https://my-resource.services.ai.azure.com/api/projects/my-project`, the wrapper
derives the resource URL and calls `/openai/v1/decisions` directly.
Project endpoints require a token credential.

For a direct endpoint, `credential` also accepts an API key string or
`AzureKeyCredential`. An omitted credential defaults to `AZURE_OPENAI_API_KEY`
for direct endpoints, then `DefaultAzureCredential`. Token providers refresh
authentication for subsequent requests. You can instead pass the upstream
`api_key` parameter, including a token callable.

With no explicit endpoint, resolution uses the first configured value:

1. `AZURE_AI_PROJECT_ENDPOINT`, then `FOUNDRY_PROJECT_ENDPOINT`.
2. `AZURE_AI_OPENAI_ENDPOINT` (including `/openai/v1`).
3. `AZURE_OPENAI_ENDPOINT` (resource root; `/openai/v1` is appended).

`model` defaults to `AZURE_OPENAI_DEPLOYMENT_NAME`. Explicit constructor values
take precedence over environment variables.

## Async and composition

Use `await decisions.ainvoke(request)` or `await decisions.abatch(requests)`
for async calls. Both synchronous and asynchronous Azure token credentials are
supported; an async-only credential requires async invocation.

Inherited Runnable features include `batch`, `abatch`, callbacks, and composition
with `|`. OpenAI SDK options such as `timeout`, `max_retries`, `default_headers`,
`default_query`, `http_client`, and `http_async_client` are forwarded upstream.
The caller owns any supplied HTTP clients and Azure credentials.

See the [runnable sample](../../../samples/decisions/README.md).
