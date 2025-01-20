# langchain-azure-ai

This package contains the LangChain integration for Azure AI Foundry. To learn more about how to use this package, see the LangChain documentation in [Azure AI Foundry](https://aka.ms/azureai/langchain).

> [!NOTE]
> This package is in Public Preview. For more information, see [Supplemental Terms of Use for Microsoft Azure Previews](https://azure.microsoft.com/support/legal/preview-supplemental-terms/).

## Installation

```bash
pip install -U langchain-azure-ai
```

For using tracing capabilities with OpenTelemetry, you need to add the extras `opentelemetry`:

```bash
pip install -U langchain-azure-ai[opentelemetry]
```

## Changelog

- **0.1.1**

  - You can now create `AzureAIEmbeddingsModel` and `AzureAIChatCompletionsModel` clients directly from your AI project's connection string using the parameter `project_connection_string`. Your default Azure AI Services connection is used to find the model requested. This requires to have `azure-ai-projects` package installed.
  - Support for native LLM structure outputs. Use `with_structured_output(method="json_schema")` to use native structured schema support. Use `with_structured_output(method="json_mode")` to use native JSON outputs capabilities. By default, LangChain uses `method="function_calling"` which uses tool calling capabilities to generate valid structure JSON payloads. This requires to have `azure-ai-inference >= 1.0.0b7`.
  - Fixes an issue when using tool calling capabilities with functions. Before, the key `function` was missing from the `additional_kwargs` property of `ToolMessage`, which is used by some integration packages.

- **0.1.0**:

  - Introduce `AzureAIEmbeddingsModel` for embedding generation and `AzureAIChatCompletionsModel` for chat completions generation using the Azure AI Inference API. This client also supports GitHub Models endpoint.
  - Introduce `AzureAIInferenceTracer` for tracing with OpenTelemetry and Azure Application Insights.