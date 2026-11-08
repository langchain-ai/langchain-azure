# 🦜️🔗 LangChain Azure

Python integrations for building LangGraph, LangChain, and Deep Agents applications on Azure.
Each package is maintained independently and supports Python 3.11 through 3.14.

## Packages

Choose a package by its purpose. Package names link to their documentation;
the Get started column links to samples or quick starts with setup instructions.

| Package | Purpose | Get started |
| --- | --- | --- |
| [langchain-azure-ai](./libs/azure-ai/README.md) | Microsoft Foundry models and embeddings, existing agents as graph nodes, LangGraph hosting, tools, remote skills, MCP, memory, guardrails, tracing, and Azure AI Search. | [Foundry model quick start](#microsoft-foundry-models) · [Agent node quick start](./libs/azure-ai/README.md#microsoft-foundry-agent-service) |
| [langchain-azure-compute](./libs/azure-compute/README.md) | Run agent-generated code and provide Deep Agents compute backends with Azure Container Apps dynamic sessions and sandboxes. | [Dynamic sessions quick start](./libs/azure-compute/README.md#dynamic-sessions) · [Sandbox quick start](./libs/azure-compute/README.md#sandboxes-deep-agents-backend) |
| [langchain-azure-cosmosdb](./libs/azure-cosmosdb/README.md) | Vector search, semantic caching, and chat history; LangGraph checkpoints, node caching, and stores for data shared across threads in Azure Cosmos DB. | [Cosmos DB samples](./samples/cosmosdb-nosql/README.md) |
| [langchain-sqlserver](./libs/sqlserver/README.md) | Vector search and chat history in Azure SQL and SQL Server. | [Azure SQL samples](https://github.com/Azure-Samples/azure-sql-langchain) |
| [langchain-azure-postgresql](./libs/azure-postgresql/README.md) | Vector search in Azure Database for PostgreSQL with DiskANN indexing. | [Vector store quick start](./libs/azure-postgresql/README.md#usage) |
| [langchain-azure-storage](./libs/azure-storage/README.md) | Load documents from Azure Blob Storage and persist Deep Agents files in blob containers. | [Deep Agents samples](./samples/deepagents-storage-backend/README.md) · [Document loader quick start](./libs/azure-storage/README.md#azure-blob-storage-document-loader-usage) |
| [langchain-azure-dynamic-sessions](./libs/azure-dynamic-sessions/README.md) | **Deprecated.** Azure Container Apps dynamic sessions integrations, replaced by `langchain-azure-compute`. | [Migration instructions](./libs/azure-dynamic-sessions/README.md) |

## Integrations

Browse by integration type or agent capability:

- **Chat models and embedding models**: [Foundry model quick start](#microsoft-foundry-models) and [embedding model guide](https://docs.langchain.com/oss/python/integrations/providers/azure_ai#embedding-models).
- **Tools and middleware**: [Document Intelligence sample](./samples/react-agent-docintelligence/README.md) and [Content Safety guardrails sample](./samples/unified-moderation/basic_usage.py).
- **Remote skills**: Access agent skills managed in [Foundry Toolbox](https://reference.langchain.com/python/langchain-azure-ai/tools/_toolbox/AzureAIProjectToolbox).
- **MCP**: Load managed MCP tools from Foundry Toolbox with [AzureAIProjectToolbox.get_tools](https://reference.langchain.com/python/langchain-azure-ai/tools/_toolbox/AzureAIProjectToolbox/get_tools).
- **Document loaders**: [Content Understanding guide](./libs/azure-ai/README.md#microsoft-foundry-tools) and [Azure Blob Storage quick start](./libs/azure-storage/README.md#azure-blob-storage-document-loader-usage).
- **Vector stores**: [Azure AI Search sample](./samples/rag-storage-document-loaders/README.md), [Azure Cosmos DB samples](./samples/cosmosdb-nosql/README.md#langchain-integrations), [Azure SQL and SQL Server samples](https://github.com/Azure-Samples/azure-sql-langchain), and [Azure Database for PostgreSQL quick start](./libs/azure-postgresql/README.md#usage).
- **Checkpointers and stores**: Persist graph execution state or store data shared across threads. See [Foundry State Store checkpoints](./libs/azure-ai/README.md#durable-graph-state) and [Cosmos DB checkpointer and store samples](./samples/cosmosdb-nosql/README.md#langgraph-integrations).
- **Memory**: Extract and recall information from conversations across sessions. See the [Foundry memory sample](./samples/foundry-memory/README.md).
- **Sandboxes and filesystem backends**: [Azure Container Apps compute quick starts](./libs/azure-compute/README.md) and [Deep Agents file samples with Azure Blob Storage](./samples/deepagents-storage-backend/README.md).
- **Callbacks and tracing**: [Azure Application Insights sample](./samples/enable_auto_tracing_appinsights.py).

## LangGraph on Azure

### Microsoft Foundry Agent Service

Use `AgentServiceFactory.get_agent_node` to reference an agent that already exists
in Microsoft Foundry and use it as a node in your LangGraph graph. The agent runs
in Foundry and can be built independently of LangChain or LangGraph, using the
Foundry portal or other SDKs. See the [Agent node quick start](./libs/azure-ai/README.md#microsoft-foundry-agent-service).

### Checkpoints and stores

Persist graph state with [FoundryCheckpointSaver backed by Foundry State Store](./libs/azure-ai/README.md#durable-graph-state)
in Foundry-hosted agents, or the [Azure Cosmos DB checkpointer](./libs/azure-cosmosdb/README.md#langgraph-checkpointer).
For data your application manages across threads, use the [Azure Cosmos DB store samples](./samples/cosmosdb-nosql/README.md#langgraph-integrations).

### Hosting applications on Foundry

To deploy a LangChain, LangGraph, or Deep Agents application to Foundry as a hosted agent,
see the [LangGraph hosting guide](https://learn.microsoft.com/azure/foundry/how-to/develop/langchain-hosted-agents).

## Quick Start with langchain-azure-ai

### Install langchain-azure-ai

```bash
pip install -U langchain-azure-ai
az login
```

### Microsoft Foundry Models

This example uses the Responses API through LangChain's chat model interface.
Use a Microsoft Foundry project with a model deployment that supports the Responses API.
Your identity needs the **Azure AI Developer** role on the project.
Replace the project endpoint and model deployment name below.

```python
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

model = AzureAIOpenAIApiChatModel(
    project_endpoint=(
        "https://{your-resource-name}.services.ai.azure.com"
        "/api/projects/{your-project}"
    ),
    credential=DefaultAzureCredential(),
    model="{your-model-deployment-name}",
    use_responses_api=True,
)

model.invoke("Translate 'Hello!' into Italian.").pretty_print()
```

For endpoint and authentication options, see the [model quick start](./libs/azure-ai/README.md#microsoft-foundry-models)
and the [LangChain and LangGraph development guide](https://aka.ms/azureai/langchain).

## Samples

Explore the [samples](./samples) for database, memory, storage, tools, guardrails, and tracing integrations.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.
