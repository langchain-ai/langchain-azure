# langchain-azure-ai

This package contains the LangChain integration for Azure AI Inference capabilities in Azure AI Foundry. To learn more about how to use this package, see the LangChain documentation in [Azure AI Foundry](https://aka.ms/azureai/langchain).

## Installation

```bash
pip install -U langchain-azure-ai
```

For using tracing capabilities with OpenTelemetry, you need to add the extras `opentelemetry`:

```bash
pip install -U langchain-azure-ai[opentelemetry]
```

## Changelog

- **0.1.0**:

  - Introduce `AzureAIEmbeddingsModel` for embedding generation and `AzureAIChatCompletionsModel` for chat completions generation using the Azure AI Inference API. This clients also support GitHub Models endpoint.
  - Introduce `AzureAIInferenceTracer` for tracing with OpenTelemetry and Azure Application Insights.