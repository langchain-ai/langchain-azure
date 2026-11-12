# Azure OpenAI Decisions

## Scope

Expose the [Foundry Azure OpenAI Decisions API](https://learn.microsoft.com/azure/foundry/openai/how-to/decisions)
through `langchain-azure-ai`. This integration targets `/openai/v1/decisions`,
currently using a `gpt-6-luna` deployment. Microsoft-Decision-1 uses the separate
`/providers/microsoft/v1/systemone` protocol and is outside this design's scope.

## Interface

`AzureAIOpenAIApiDecisions` is a `Runnable[DecisionRequest, DecisionResponse]`.
It accepts shared input and named questions, and returns typed answers keyed by
question name:

- `Predicate`: the probability that a condition is true.
- `Choice`: a selection from fixed options, with probabilities and confidence.
- `Score`: a rating over ordered levels, with probabilities and confidence.

A question can return a refusal while other questions receive answers. The
result preserves these answer types rather than converting them to `AIMessage`.
This contract is why Decisions has a dedicated Runnable instead of extending
the chat model interface.

## Adapter

The class extends `langchain_openai.decisions.OpenAIDecisions` and reexports its
question and answer types from `langchain_azure_ai.decisions`. Request building
and response parsing stay upstream; the adapter adds Azure configuration using
the existing endpoint and credential helpers. This requires
`langchain-openai>=1.7.2`.

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
