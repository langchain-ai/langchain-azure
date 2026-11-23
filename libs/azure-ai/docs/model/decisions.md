# Azure OpenAI Decisions

## Scope

Expose the [Foundry Azure OpenAI Decisions API](https://learn.microsoft.com/azure/foundry/openai/how-to/decisions)
through `langchain-azure-ai`. This integration targets `/openai/v1/decisions`,
currently using a `gpt-6-luna` deployment. Microsoft-Decision-1 uses the separate
`/providers/microsoft/v1/systemone` protocol and is outside this design's scope.

## Interface

`AzureAIOpenAIApiDecisionsClassifier` implements LangChain's `BaseDecisionModel`, a
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

## Adapter

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

## Stability

The upstream Runnable is beta, so the Azure adapter uses the existing
`@experimental(...)` decorator and emits `ExperimentalWarning` on construction.
No additional runtime opt-in flag is introduced.

Use `AzureAIOpenAIApiDecisionsClassifier` in new code. The previous name,
`AzureAIOpenAIApiDecisions`, remains an alias at both existing import paths.
LangChain serialization keeps the original class ID so existing serialized
Runnables can still be loaded with either name in `allowed_objects`.
