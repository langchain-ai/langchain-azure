# Foundry Decision Models

## Choose an adapter

| Models | Adapter | API path |
| --- | --- | --- |
| Compatible Microsoft Decision models, verified with `Microsoft-Decision-1` and `Microsoft-Decision-1.1` | `AzureAIMicrosoftDecisions` | `/providers/microsoft/v1/systemone` |
| `gpt-6-luna` | `AzureAIOpenAIApiDecisions` | `/openai/v1/decisions` |

## Shared interface

Both adapters implement LangChain's `BaseDecisionModel`, a
`Runnable[ClassifierRequest, ClassifierResponse]`. Call `invoke` or `ainvoke`
with `state` and a map of named questions. Read answers by question name:

- `Noul`: probability that a condition is true, in `result.nouls`.
- `Choice`: selection from fixed options, in `result.choices`.
- `Score`: rating over ordered levels, in `result.scores`.

`Choice` and `Score` answers include probabilities and confidence. `Choice`
requires at least two labels. Neither adapter supports `Box` questions.

## Example

Set `AZURE_AI_PROJECT_ENDPOINT` to the Foundry project URL and
`DECISION_DEPLOYMENT_NAME` to your deployment name, not the underlying model
name. Both adapters derive the resource URL from the project URL and default to
`DefaultAzureCredential` in this configuration.

To switch between compatible Microsoft Decision deployments, change only
`DECISION_DEPLOYMENT_NAME`. For a `gpt-6-luna` deployment, also replace
`AzureAIMicrosoftDecisions` with `AzureAIOpenAIApiDecisions` in the import and
constructor. The request and response code stays the same.

```python
import os

from langchain_azure_ai.decisions import AzureAIMicrosoftDecisions, Choice

decisions = AzureAIMicrosoftDecisions(
    project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    model=os.environ["DECISION_DEPLOYMENT_NAME"],
)

result = decisions.invoke(
    {
        "state": {"ticket": "I was charged twice for this order."},
        "questions": {
            "team": Choice(
                instructions="Which team should handle this ticket?",
                criteria={"billing": "Payments and refunds", "support": "Other requests"},
            )
        },
    }
)
print(result.choices["team"].choice)
```

## Adapter differences

### Microsoft Decision

`AzureAIMicrosoftDecisions` accepts text, JSON objects, and JSON arrays. It
does not convert LangChain messages or send image assets. To use an API key,
pass it as `credential`. To skip project URL resolution, pass the Foundry
resource URL as `endpoint`. See the [Microsoft Decision model guide](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision).

### Azure OpenAI Decisions

`AzureAIOpenAIApiDecisions` uses the beta `OpenAIClassifier` from
`langchain-openai>=1.8.0` and emits `ExperimentalWarning` on construction. The
classifier maps `Noul` to `Predicate`. Root messages can carry supported
images. Media embedded in JSON state is dropped with a warning. A refusal on
any question raises `ModelRefusalError` for the whole request.

`project_endpoint` requires an Entra credential. To use an API key, pass a
resource `endpoint` ending in `/openai/v1` and the key as `credential`. The
adapter retains the resolved `base_url` and `api_key` for Runnable
reconstruction and serialization, using the upstream secret-reference
mechanism for credentials. See the [Azure OpenAI Decisions guide](https://learn.microsoft.com/azure/foundry/openai/how-to/decisions).
