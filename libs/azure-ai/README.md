# langchain-azure-ai

This package contains the LangChain integration for Microsoft Foundry (formerly known Azure AI). To learn more about how to use this package, see the LangChain documentation in [Microsoft Foundry](https://aka.ms/azureai/langchain).

## Installation

```bash
pip install -U langchain-azure-ai
```

For using tools, including Azure AI Document Intelligence, Azure AI Text Analytics for Health, or Azure LogicApps, please install the extras `tools`:

```bash
pip install -U langchain-azure-ai[tools]
```

For using tracing capabilities with OpenTelemetry, you need to add the extras `opentelemetry`:

```bash
pip install -U langchain-azure-ai[opentelemetry]
```

For hosting LangGraph agents on Microsoft Foundry with the Responses or Invocations protocols, install the `hosting` extra:

```bash
pip install -U langchain-azure-ai[hosting]
```

If you are transitioning from Microsoft Foundry classic and you need access to deprecated classes, use `[v1]` extra.

```bash
pip install -U langchain-azure-ai[v1]
```

## Quick Start with langchain-azure-ai

The `langchain-azure-ai` package uses the Microsoft Foundry family of SDKs and client libraries for Azure to provide first-class support of Microsoft Foundry capabilities in LangChain and LangGraph.

This package includes:

* [Microsoft Foundry Models inference](https://github.com/langchain-ai/langchain-azure/libs/azure-ai/langchain_azure_ai/chat_models)
* [Microsoft Foundry Tools](https://github.com/langchain-ai/langchain-azure/libs/azure-ai/langchain_azure_ai/tools)
* [Microsoft Foundry Content Safety](https://github.com/langchain-ai/langchain-azure/libs/azure-ai/langchain_azure_ai/agents/middleware)
* [Microsoft Foundry Agent Service](https://github.com/langchain-ai/langchain-azure/libs/azure-ai/langchain_azure_ai/agents)
* [LangGraph hosting for Microsoft Foundry](https://github.com/langchain-ai/langchain-azure/tree/main/libs/azure-ai/langchain_azure_ai/agents/hosting)
* [Azure AI Search](https://github.com/langchain-ai/langchain-azure/libs/azure-ai/langchain_azure_ai/vectorstores)
* [Microsoft Foundry Tools](https://github.com/langchain-ai/langchain-azure/tree/main/libs/azure-ai/langchain_azure_ai/tools) (including Azure AI Content Understanding, Document Intelligence, and more)

Here's a quick start example to show you how to get started with the Chat Completions model. For more details and tutorials see [Get started with LangChain and LangGraph with Foundry](https://aka.ms/azureai/langchain).

### Microsoft Foundry Models

```python
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_core.messages import HumanMessage, SystemMessage

model = AzureAIOpenAIApiChatModel(
    endpoint="https://{your-resource-name}.services.ai.azure.com/openai/v1",
    credential="your-api-key", #if using Entra ID you can should use DefaultAzureCredential() instead
    model="gpt-5"
)

messages = [
    SystemMessage(
      content="Translate the following from English into Italian"
    ),
    HumanMessage(content="hi!"),
]

model.invoke(messages).pretty_print()
```

```output
================================== Ai Message ==================================
Ciao!
```

You can also use builtin tools with them:

```python
from langchain_azure_ai.tools.builtin import ImageGenerationTool

model_with_image_gen = model.bind_tools([ImageGenerationTool(model="gpt-image-1.5", size="1024x1024")])
result = model_with_image_gen.invoke(
    "Generate an image based on the following description: A futuristic cityscape at sunset with flying cars and neon lights."
)
```

Models in Microsoft Foundry Models are OpenAI-compatible and can be used with the class:

```python
model = AzureAIOpenAIApiChatModel(
    endpoint="https://{your-resource-name}.services.ai.azure.com/openai/v1",
    credential="your-api-key",
    model="Mistral-Large-3"
)
```

### Microsoft Foundry Agent Service

Compose complex graphs by using agents running in the Agent Service:

```python
from azure.identity import DefaultAzureCredential
from langchain_core.messages import AIMessage, HumanMessage
from langchain_azure_ai.agents import AgentServiceFactory
from langchain_azure_ai.utils.agents import pretty_print

factory = AgentServiceFactory(
    project_endpoint="https://{your-resource-name}.services.ai.azure.com/api/projects/{your-project}",
    credential=DefaultAzureCredential()
)

echo_node = factory.get_agent_node(name="my-echo-agent", version="latest")
```

Agent Service nodes run in Microsoft Foundry but can be added to any graph:

```python
graph.add_node("expert_node", echo_node)
```

Use the graph as usual:

```python
agent = graph.compile()
messages = [HumanMessage(content="I'm a genius and I love programming!")]
response = agent.invoke({"messages": messages})

pretty_print(response)
```

```output
================================ Human Message =================================

I'm a genius and I love programming!
================================== Ai Message ==================================
Name: my-echo-agent

You're not a genius and you don't love programming!
```

### Hosting LangGraph agents on Microsoft Foundry

Install the hosting extra to expose a compiled LangGraph graph through Foundry-compatible protocols:

```bash
pip install -U langchain-azure-ai[hosting]
```

```python
from langchain_azure_ai.agents.hosting import ResponsesHostServer

graph = build_my_graph()  # returns a compiled LangGraph graph

if __name__ == "__main__":
    ResponsesHostServer(graph).run()
```

`ResponsesHostServer` serves the OpenAI Responses-style `/responses` endpoint. `InvocationsHostServer` serves the generic `/invocations` endpoint for applications that want to define their own JSON request and response shape.

The Responses host uses one conversation-state source per graph. The policy depends on whether the hosted graph has a LangGraph checkpointer:

| Graph configuration | Conversation source | Graph input on later turns |
|---|---|---|
| Graph compiled with a checkpointer | LangGraph checkpoint state keyed by the conversation/thread id | Current request input only |
| Graph without a checkpointer | Responses transcript history from the underlying response provider | Prior Responses history plus current input |

Checkpointed multi-turn conversations must use an explicit `conversation.id`
or run with `steerable_conversations=True`. Both modes are linear and preserve
the latest LangGraph checkpoint through Agent Server's public
`conversation_chain_metadata` API. Forking with only `previous_response_id`
when steering is disabled is not supported for checkpointed graphs.

The Responses transcript provider is selected by the underlying `azure-ai-agentserver-responses` runtime. Local runs and tests use an in-memory provider by default. Foundry-hosted containers use the Foundry-backed storage provider when the platform environment variables are present. This transcript store is separate from the LangGraph checkpointer, which stores graph runtime state.


### Auto tracing to Azure Application Insights

To trace your LangChain / LangGraph applications with Azure Application Insights, first install the OpenTelemetry extras:

```bash
pip install -U langchain-azure-ai[opentelemetry]
```

Then enable auto tracing in your application. Every `BaseCallbackManager` and LangGraph callback manager created after this call will automatically include the Azure tracer:

```python
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing

enable_auto_tracing(
    connection_string="<your-application-insights-connection-string>",
    auto_configure_azure_monitor=True,
    enable_content_recording=False,      # set to True to capture message payloads
    provider_name="azure.ai.openai",
    trace_all_langgraph_nodes=True,
)
```

For a complete end-to-end example, see [`samples/enable_auto_tracing_appinsights.py`](../../samples/enable_auto_tracing_appinsights.py).

### Microsoft Foundry Tools

Use tools from Azure AI services as LangChain tools via `AzureAIServicesToolkit`. Available tools include Azure AI Content Understanding, Document Intelligence, Image Analysis, Text Analytics for Health, and more.

Azure AI Content Understanding is also available as a document loader via `AzureAIContentUnderstandingLoader`. See the [Content Understanding loader notebook](./docs/content_understanding_loader_demo.ipynb) for a full walkthrough.


## Changelog

- **1.2.8**:

  - We extended `AzureAIProjectToolbox` to forward the `x-agent-foundry-call-id` header on toolbox calls, ensuring per-call isolation and authorization on the toolbox side for deployed agents (hosted agent container spec 2.0.0). [#785](https://github.com/langchain-ai/langchain-azure/pull/785)
  - We fixed an issue where the LangGraph hosting wrapper for Microsoft Foundry (`langchain_azure_ai.agents.hosting`) did not stream model reasoning summaries. Hosted agents configured with reasoning summaries now emit them through the streaming Responses output. [#771](https://github.com/langchain-ai/langchain-azure/pull/771)

- **1.2.7**:

  - **[NEW]** We extended `AzureAIProjectToolbox` with skills support: new `get_skills`/`aget_skills` methods load toolbox skills as DeepAgents-ready file mappings, and new `get_resources`/`aget_resources` methods fetch toolbox resources (such as skills) as LangChain `Blob` objects with flexible URI-scheme filtering. [#700](https://github.com/langchain-ai/langchain-azure/pull/700)

- **1.2.6**:

  - We added a hosting User-Agent stamp on OpenAI and Anthropic SDK clients used by LangGraph agents hosted in Microsoft Foundry, so outbound model calls from hosted agents are correctly attributed to `langchain-azure-ai/hosting`. [#671](https://github.com/langchain-ai/langchain-azure/pull/671)
  - We updated dependencies including `aiohttp` and the `uv` group to incorporate maintenance and security updates. [#654](https://github.com/langchain-ai/langchain-azure/pull/654) [#657](https://github.com/langchain-ai/langchain-azure/pull/657) [#663](https://github.com/langchain-ai/langchain-azure/pull/663) [#667](https://github.com/langchain-ai/langchain-azure/pull/667)

- **1.2.5**:

  - We added `messages_key` configurability in `AzureAIMemoryMiddleware` to support agent states that use non-default message field names. [#647](https://github.com/langchain-ai/langchain-azure/pull/647)
  - We improved Azure AI Memory and content safety middleware docstrings with clearer examples, and fixed memory retriever output context formatting for better usability. [#647](https://github.com/langchain-ai/langchain-azure/pull/647)

- **1.2.4**:

  - **[NEW]** We introduced support for hosting LangGraph agents in Microsoft Foundry through `ResponsesHostServer` and `InvocationsHostServer`, including checkpoint integration for conversation state. [#595](https://github.com/langchain-ai/langchain-azure/pull/595)
  - **[NEW]** We introduced Azure AI Memory turn-sync middleware and an on-demand memory retrieval tool to improve multi-turn memory workflows. [#637](https://github.com/langchain-ai/langchain-azure/pull/637)
  - We fixed duplicate conversation history handling and improved parallel tool call handling in hosted Responses flows. [#626](https://github.com/langchain-ai/langchain-azure/pull/626)
  - We updated Azure AI Content Understanding tooling to use the SDK `to_llm_input` helper for improved request shaping. [#554](https://github.com/langchain-ai/langchain-azure/pull/554)
  - We introduced `AzureAIProjectToolbox.get_tools_requiring_approval` to help identify tools that require explicit user approval in agent flows. [#537](https://github.com/langchain-ai/langchain-azure/pull/537)
  - We updated key dependencies such as `urllib3`, `idna`, and `starlette` to incorporate maintenance and security updates. [#592](https://github.com/langchain-ai/langchain-azure/pull/592) [#582](https://github.com/langchain-ai/langchain-azure/pull/582) [#616](https://github.com/langchain-ai/langchain-azure/pull/616) [#623](https://github.com/langchain-ai/langchain-azure/pull/623)

- **1.2.3**:

  - **[NEW]** We introduced `AzureAIProjectToolbox` for accessing tools managed in Azure AI Foundry projects, simplifying tool configuration and management. [#525](https://github.com/langchain-ai/langchain-azure/pull/525)
  - **[NEW]** We migrated speech-to-text tool for Azure AI from the community package, enabling speech integration capabilities. [#499](https://github.com/langchain-ai/langchain-azure/pull/499)
  - We updated `AzureAIOpenTelemetryTracer` to emit OpenTelemetry spec-compliant spans for GenAI operations. [#509](https://github.com/langchain-ai/langchain-azure/pull/509)
  - We patched 21 security alerts across dependency packages including `langsmith`, `cryptography`, `langchain-core`, and dev dependencies to improve package security posture. [#496](https://github.com/langchain-ai/langchain-azure/pull/496)

- **1.2.2**:

  - **[NEW]** We introduced `AzureAIContentUnderstandingLoader` document loader for extracting content from documents, images, audio, and video using Azure AI Content Understanding. [#423](https://github.com/langchain-ai/langchain-azure/pull/423)
  - **[NEW]** We introduced `AzureAIContentUnderstandingTool` for using Content Understanding as an agent tool, also available via `AzureAIServicesToolkit`. [#446](https://github.com/langchain-ai/langchain-azure/pull/446)
  - We introduced `context_extractor` support across content safety middleware classes, so you can control how content is extracted from agent state before safety checks run. [#419](https://github.com/langchain-ai/langchain-azure/pull/419)
  - We introduced `context_extractor` support for `AzureGroundednessMiddleware` and added a notebook example for easier adoption. [#410](https://github.com/langchain-ai/langchain-azure/pull/410)
  - We changed the default implementation of `init_chat_model("azure_ai:<your-model>")` to use the OpenAI Responses API path for improved compatibility with modern LangChain chat model initialization. [#409](https://github.com/langchain-ai/langchain-azure/pull/409)
  - We fixed an `AttributeError` in `AzureAIOpenTelemetryTracer.on_chain_start` when chain inputs were not dictionaries. [#317](https://github.com/langchain-ai/langchain-azure/pull/317)
  - We upgraded the `requests` dependency for this package to include upstream security and maintenance updates. [#417](https://github.com/langchain-ai/langchain-azure/pull/417)
  - We patched multiple high-severity dependency vulnerabilities (including `PyJWT`, `orjson`, and `tornado`) to improve package security posture. [#412](https://github.com/langchain-ai/langchain-azure/pull/412)

- **1.2.1**:

    - You can now use `context_extractor` argument in classes `langchain_azure_ai.agents.middleware.` to configre how middleware instract extract content from your state.
    - We changed the default implementation of `init_chat_model("azure_ai:<your-model>")` to use OpenAI Responses API (this is also the default if using `langchain>=1.2.3`).

- **1.2.0**:

    - We now require `langchain>=1.2` so our streaming implementation matches the latest version of `langchain`.
    - We introduced `langchain_azure_ai.agents.middleware.content_safety.*` namespace which unlocks the power of Azure AI Content Safety with LangChain.
    - We introduced `langchain_azure_ai.tools.builtin.*` namespace with server-side tools that can be used for models running in Microsoft Foundry.
    - We fixed an issue with duplicated spans generated in OpenTelemetry tracer. [#398](https://github.com/langchain-ai/langchain-azure/pull/398).
    - We fixed an issue in `init_embeddings(provider="azure_ai")` where an incorrect kwarg was passed.

- **1.1.0**:

    - Creating agents using Foundry Agents V1 has been deprecated in favor of V2. `langchain_azure_ai.agents.AgentServiceFactory` now using V2 implementation. Namespace `langchain_azure_ai.agents.v1.AgentServiceFactory` is marked as deprecated and requires the extra `v1` to be used.
    - Chat and embedding models using Azure AI Inference SDK has been deprecated in favor of OpenAI-compatible APIs. Namespace `langchain_azure_ai.chat_models.inference.AzureAIChatCompletionsModel` and `langchain_azure_ai.embeddings.inference.AzureAIEmbeddingsModel` are marked as deprecated and require the extra `v1` to be used.

- **1.0.62**:

    - We introduced support for asynchhronous agents operation and tracing using our OpenTelemetry tracer for context to propagate correctly. [#290].(https://github.com/langchain-ai/langchain-azure/pull/290).
    - We introduced support for Bash operations in `langchain-azure-dynamic-session`. [#238](https://github.com/langchain-ai/langchain-azure/pull/238).
    - We introduced support for Agent Service V2 in Microsoft Foundry. [PR #257](https://github.com/langchain-ai/langchain-azure/pull/257).
    - We added a new tool to generate images based on OpenAI-compatible image generation models. [PR #325](https://github.com/langchain-ai/langchain-azure/pull/325)
    - We fixed an issue when `on_tool_start` ignores `enable_content_recording`. Now it doesn't. [#261](https://github.com/langchain-ai/langchain-azure/pull/261).
    - We fixed a problem when uploaded files were not considered by the `CodeInterpreterTool` for the Agent Service. [#256](https://github.com/langchain-ai/langchain-azure/pull/256).
    - We fixed an issue when using `AzureAIOpenTelemetryTracer` on a Mac. [#234](https://github.com/langchain-ai/langchain-azure/pull/234).

- **1.0.61**:

    - This release reverts the code to the state of v1.0.5 while updating the version number to 1.0.61.

- **1.0.5**:

    - We fixed an issue with the content type of messages in `AzureAIChatCompletionsModel`. See [PR #245].
    - We improve metadata generated for `AzureAIOpenTelemetryTracer`. See [PR ##233].

- **1.0.4**:

    - We fixed an issue with dependencies resolution for `azure-ai-agents` where the incorrect version was picked up. See [PR #221].
    - We fixed an issue with `AzureAIOpenTelemetryTracer` where spans context was not correctly propagated when called from another service. See [PR #217].
    - We fixed an issue where `AzureAIOpenTelemetryTracer` where context was deallocated incorrectly, preventing tools like `langdev` to correctly emit traces. See [Issue #212].
    - We introduced improvements in the order in which environment variables `AZURE_AI_*` are read.
    - Internal: We improved `AzureAIOpenTelemetryTracer` test coverage. See [PR #239](https://github.com/langchain-ai/langchain-azure/pull/239).

- **1.0.2**:

    - We updated the `AzureAIOpenTelemetryTracer` to create a parent trace for multi agent scenarios. Previously, you were required to do this manually, which was unnecesary.

- **1.0.0**:

    - We introduce support for LangChain and LangGraph 1.0.

- **0.1.8**:

    - We fixed some issues with `AzureAIOpenTelemetryTracer`, including compliant hierarchy, tool spans under chat, finish reason normalization, conversation id. See [PR #167]
    - We fixed an issue with taking image inputs for declarative agents created with Azure AI Foundry Agents service.
    - We enhanced tool descriptions to improve tool call accuracy. 

- **0.1.7**:

  - **[NEW]**: We introduce LangGraph support for declarative agents created in Azure AI Foundry. You can now compose complex graphs in LangGraph and add nodes that take advantage of Azure AI Agent Service. See [`AgentServiceFactory`](./langchain_azure_ai/agents/agent_service.py#L44)
  - We fix an issue with the interface of `AzureAIEmbeddingsModel` [#158](https://github.com/langchain-ai/langchain-azure/issues/158).
  - We improve the signatures of the tools `AzureAIDocumentIntelligenceTool`, `AzureAIImageAnalysisTool`, and `AzureAITextAnalyticsHealthTool` [PR #160](https://github.com/langchain-ai/langchain-azure/pull/160).

- **0.1.6**:

  - **[Breaking change]:** Using parameter `project_connection_string` to create `AzureAIEmbeddingsModel` and `AzureAIChatCompletionsModel` is not longer supported. Use `project_endpoint` instead.
  - **[Breaking change]:** Class `AzureAIInferenceTracer` has been removed in favor of `AzureAIOpenTelemetryTracer` which has a better support for OpenTelemetry and the new semantic conventions for GenAI.
  - Adding the following tools to the package: `AzureAIDocumentIntelligenceTool`, `AzureAIImageAnalysisTool`, and `AzureAITextAnalyticsHealthTool`. You can also use `AIServicesToolkit` to have access to all the tools in Azure AI Services.

- **0.1.4**:

  - Bug fix [#91](https://github.com/langchain-ai/langchain-azure/pull/91).

- **0.1.3**:

  - **[Breaking change]:** We renamed the parameter `model_name` in `AzureAIEmbeddingsModel` and `AzureAIChatCompletionsModel` to `model`, which is the parameter expected by the method `langchain.chat_models.init_chat_model`.
  - We fixed an issue with JSON mode in chat models [#81](https://github.com/langchain-ai/langchain-azure/issues/81).
  - We fixed the dependencies for NumpPy [#70](https://github.com/langchain-ai/langchain-azure/issues/70).
  - We fixed an issue when tracing Pyndantic objects in the inputs [#65](https://github.com/langchain-ai/langchain-azure/issues/65).
  - We made `connection_string` parameter optional as suggested at [#65](https://github.com/langchain-ai/langchain-azure/issues/65).

- **0.1.2**:

  - Bug fix [#35](https://github.com/langchain-ai/langchain-azure/issues/35).

- **0.1.1**: 

  - Adding `AzureCosmosDBNoSqlVectorSearch` and `AzureCosmosDBNoSqlSemanticCache` for vector search and full text search.
  - Adding `AzureCosmosDBMongoVCoreVectorSearch` and `AzureCosmosDBMongoVCoreSemanticCache` for vector search.
  - You can now create `AzureAIEmbeddingsModel` and `AzureAIChatCompletionsModel` clients directly from your AI project's connection string using the parameter `project_connection_string`. Your default Azure AI Services connection is used to find the model requested. This requires to have `azure-ai-projects` package installed.
  - Support for native LLM structure outputs. Use `with_structured_output(method="json_schema")` to use native structured schema support. Use `with_structured_output(method="json_mode")` to use native JSON outputs capabilities. By default, LangChain uses `method="function_calling"` which uses tool calling capabilities to generate valid structure JSON payloads. This requires to have `azure-ai-inference >= 1.0.0b7`.
  - Bug fix [#18](https://github.com/langchain-ai/langchain-azure/issues/18) and [#31](https://github.com/langchain-ai/langchain-azure/issues/31).

- **0.1.0**:

  - Introduce `AzureAIEmbeddingsModel` for embedding generation and `AzureAIChatCompletionsModel` for chat completions generation using the Azure AI Inference API. This client also supports GitHub Models endpoint.
  - Introduce `AzureAIOpenTelemetryTracer` for tracing with OpenTelemetry and Azure Application Insights.
