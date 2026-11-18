# Foundry Decision Models

Both adapters use LangChain's `BaseDecisionModel` interface with `invoke` and
`ainvoke`. Select the adapter for your deployment:

| Deployment | Adapter | API path |
| --- | --- | --- |
| Microsoft Decision deployments using the System One API | `AzureAIMicrosoftDecisions` | `/providers/microsoft/v1/systemone` |
| `gpt-6-luna` | `AzureAIOpenAIApiDecisions` | `/openai/v1/decisions` |

## Microsoft Decision

Use this adapter with a Microsoft Decision deployment that uses a compatible
System One request and response format. Set `model` to the Foundry deployment
name and `project_endpoint` to the Foundry project URL. The adapter derives the
resource URL and calls the resource-level Decision API. It defaults to
`DefaultAzureCredential`; pass `credential` explicitly to use an API key or
another Entra credential.

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

The adapter supports `Noul`, `Choice`, and `Score` questions and preserves the
probabilities and confidence reported by the model. It accepts text, JSON
objects, and JSON arrays. LangChain message conversion, image assets, and `Box`
are not supported.

Verified with `Microsoft-Decision-1` and `Microsoft-Decision-1.1`.

See the [Foundry model guide](https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision)
for deployment and API details.

## Azure OpenAI Decisions

### Scope

Expose the [Foundry Azure OpenAI Decisions API](https://learn.microsoft.com/azure/foundry/openai/how-to/decisions)
through `langchain-azure-ai`. This integration targets `/openai/v1/decisions`,
currently using a `gpt-6-luna` deployment.

### Interface

`AzureAIOpenAIApiDecisions` implements LangChain's `BaseDecisionModel`, a
`Runnable[ClassifierRequest, ClassifierResponse]`. It accepts `state` and named
Core questions, and returns typed answers keyed by
question name:

- `Noul`: the probability that a condition is true.
- `Choice`: a selection from fixed options, with probabilities and confidence.
- `Score`: a rating over ordered levels, with probabilities and confidence.

Answers use `response.nouls`, `response.choices`, and `response.scores`. This
contract supports consumers of `BaseDecisionModel` while keeping Decisions
separate from the chat model's `AIMessage` contract.

The upstream adapter defines the capability limits: `Box` is unsupported,
`Choice` needs at least two labels, and any refusal raises `ModelRefusalError`
for the entire request. Structured JSON state cannot preserve media; upstream
drops embedded media with a warning. Root messages can carry supported images.

### Adapter

The class extends `langchain_openai.decisions.OpenAIClassifier` and reexports
Core question and answer types from `langchain_azure_ai.decisions`. The upstream
classifier converts Core requests and responses to and from the OpenAI protocol,
including `Noul` to `Predicate`. The Azure adapter adds configuration using the
existing endpoint and credential helpers. This requires
`langchain-openai>=1.8.0`, which requires `langchain-core>=1.8.0`.

Azure constructor options are normalized to the inherited `base_url` and `api_key`
fields. Only this resolved configuration is retained for Runnable reconstruction
and serialization; credentials use the upstream secret-reference mechanism.

Requests go directly to the resource's `/openai/v1/decisions` endpoint. When
configured with a Foundry project endpoint, the adapter derives the resource
URL; it does not call the project or Agent Service APIs.

### Stability

The upstream `OpenAIClassifier` is beta, so `AzureAIOpenAIApiDecisions` emits
`ExperimentalWarning` on construction. No runtime opt-in flag is required.
