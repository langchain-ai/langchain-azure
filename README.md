# 🦜️🔗 LangChain Azure

Python integrations for building LangGraph, LangChain, and Deep Agents applications on Azure.
Each package is maintained independently and supports Python 3.11 through 3.14.

## Packages

Choose a package by its purpose. Package names link to their documentation;
the Get started column links to quick starts or samples with setup instructions.

| Package | Purpose | Get started |
| --- | --- | --- |
| [langchain-azure-ai](./libs/azure-ai/README.md) | Microsoft Foundry models and agents; LangGraph hosting, guardrails, tracing, tools, and Azure AI Search. | [Chat model quick start](#microsoft-foundry-models) · [Hosted-agent samples](./samples/hosting/langgraph-hosted-agents/README.md#start-here) |
| [langchain-azure-compute](./libs/azure-compute/README.md) | Run agent-generated code and provide Deep Agents compute backends with Azure Container Apps dynamic sessions and sandboxes. | [Dynamic sessions quick start](./libs/azure-compute/README.md#dynamic-sessions) · [Sandbox quick start](./libs/azure-compute/README.md#sandboxes-deep-agents-backend) |
| [langchain-azure-cosmosdb](./libs/azure-cosmosdb/README.md) | Vector search, semantic caching, and chat history; LangGraph checkpoints, node caching, and long-term memory in Azure Cosmos DB. | [Cosmos DB samples](./samples/cosmosdb-nosql/README.md) |
| [langchain-sqlserver](./libs/sqlserver/README.md) | Vector search and chat history in Azure SQL and SQL Server. | [Azure SQL samples](https://github.com/Azure-Samples/azure-sql-langchain) |
| [langchain-azure-postgresql](./libs/azure-postgresql/README.md) | Vector search in Azure Database for PostgreSQL with Microsoft Entra ID authentication and DiskANN indexing. | [Vector store quick start](./libs/azure-postgresql/README.md#usage) |
| [langchain-azure-storage](./libs/azure-storage/README.md) | Load documents from Azure Blob Storage and persist Deep Agents files in blob containers. | [Document loader quick start](./libs/azure-storage/README.md#azure-blob-storage-document-loader-usage) · [Deep Agents samples](./samples/deepagents-storage-backend/README.md) |
| [langchain-azure-dynamic-sessions](./libs/azure-dynamic-sessions/README.md) | **Deprecated.** Azure Container Apps dynamic sessions integrations, replaced by `langchain-azure-compute`. | [Migration instructions](./libs/azure-dynamic-sessions/README.md) |

## LangGraph on Azure

Build with Azure models using the [LangChain and LangGraph development guide](https://aka.ms/azureai/langchain), or bring an existing compiled graph to Microsoft Foundry with the [LangGraph hosting guide](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents).
The [hosted-agent samples](./samples/hosting/langgraph-hosted-agents/README.md#start-here) cover local runs and deployment, including [hosting an existing graph without changing its implementation](./samples/hosting/langgraph-hosted-agents/responses/09_run/README.md).
For durable graph state, use [FoundryCheckpointSaver backed by Foundry State Store](./libs/azure-ai/README.md#durable-graph-state) in Foundry-hosted agents, or the [Azure Cosmos DB checkpointer](./libs/azure-cosmosdb/README.md#langgraph-checkpointer). Cosmos DB also provides a [long-term memory store](./libs/azure-cosmosdb/README.md#langgraph-store-long-term-memory).

## Samples

- [SDK samples](https://github.com/langchain-ai/langchain-azure/tree/main/samples) demonstrate SDK usage, including hosting, memory, and storage integrations.
- [SDK notebooks](https://github.com/langchain-ai/langchain-azure/tree/main/libs/azure-ai/docs) cover Foundry models, Microsoft Foundry Agent Service, Content Safety, and Content Understanding.
- [Foundry deployment samples](https://github.com/microsoft-foundry/foundry-samples/tree/main/samples/python/hosted-agents/langgraph) demonstrate platform configuration and deployment with LangGraph.

## Quick Start with langchain-azure-ai

The `langchain-azure-ai` package uses the Microsoft Foundry family of SDKs and client libraries for Azure to provide first-class support of Microsoft Foundry capabilities in LangChain and LangGraph.

This package includes:

* [Microsoft Foundry Agent Service](./libs/azure-ai/langchain_azure_ai/agents)
* [Microsoft Foundry models](./libs/azure-ai/langchain_azure_ai/chat_models)
* [Microsoft Foundry Content Safety](./libs/azure-ai/langchain_azure_ai/agents/middleware)
* [Microsoft Foundry Tools](./libs/azure-ai/langchain_azure_ai/tools)
* [Azure AI Search](./libs/azure-ai/langchain_azure_ai/vectorstores)

For a chat model quick start, use the example below. For more details, see the [LangChain and LangGraph development guide for Microsoft Foundry](https://aka.ms/azureai/langchain).

### Install langchain-azure-ai

```bash
pip install -U langchain-azure-ai
```

### Microsoft Foundry Models

Use any Foundry Model with OpenAI-compatible APIs:

We recommend the [Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses) when your model and endpoint support it. Set `use_responses_api=True` as shown below. If you prefer Chat Completions and your model and endpoint support it, set `use_responses_api=False` instead.

```python
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_core.messages import HumanMessage, SystemMessage

model = AzureAIOpenAIApiChatModel(
    project_endpoint="https://{your-resource-name}.services.ai.azure.com/api/projects/{your-project}",
    credential=DefaultAzureCredential(), # requires Azure AI Developer role. If using keys, use parameter `endpoint` instead of `project_endpoint`.
    model="gpt-5",                       # your model deployment name
    use_responses_api=True,
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

Use the `azure_ai` provider with `init_chat_model`. Choose one of the
following authentication options.

**Option A: Project endpoint with Microsoft Entra ID**

Set `AZURE_AI_PROJECT_ENDPOINT` and sign in with `az login` for local
development:

```python
import os

from langchain.chat_models import init_chat_model

os.environ["AZURE_AI_PROJECT_ENDPOINT"] = "https://{your-resource-name}.services.ai.azure.com/api/projects/<project>"

model = init_chat_model("azure_ai:gpt-5-mini")
model.invoke("Hello!").pretty_print()
```

**Option B: OpenAI-compatible endpoint with an API key**

Set `OPENAI_API_BASE` and `OPENAI_API_KEY`. Leave `AZURE_AI_PROJECT_ENDPOINT`
unset when using this option:

```python
import os

from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_BASE"] = "https://{your-resource-name}.services.ai.azure.com/openai/v1"
os.environ["OPENAI_API_KEY"] = "{your-key}"

model = init_chat_model("azure_ai:gpt-5-mini")
model.invoke("Hello!").pretty_print()
```

### Microsoft Foundry Agent Service

Use `AgentServiceFactory` to create an agent in Microsoft Foundry Agent Service
and add it as a node in a LangGraph graph.

For local development, install the required extra and sign in:

```bash
pip install -U "langchain-azure-ai[opentelemetry]"
az login
```

Set your project endpoint and model deployment name. This creates a version
of `my-echo-agent` in your project:

```python
from azure.identity import DefaultAzureCredential
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langchain_azure_ai.agents import AgentServiceAgentState, AgentServiceFactory
from langchain_azure_ai.utils.agents import pretty_print

factory = AgentServiceFactory(
    project_endpoint="https://{your-resource-name}.services.ai.azure.com/api/projects/{your-project}",
    credential=DefaultAzureCredential()
)

echo_node = factory.create_prompt_agent_node(
    name="my-echo-agent",
    model="gpt-4.1",  # your model deployment name
    instructions="Always reply with the opposite of what the user says.",
)
```

Add Microsoft Foundry Agent Service nodes to your graph:

```python
graph = StateGraph(AgentServiceAgentState)
graph.add_node("expert_node", echo_node)
graph.add_edge(START, "expert_node")
graph.add_edge("expert_node", END)
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

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.
