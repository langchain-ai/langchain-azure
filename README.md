# 🦜️🔗 LangChain Azure

Python integrations for building LangGraph, LangChain, and Deep Agents applications on Azure.
Each package is maintained independently and supports Python 3.11 through 3.14.

## Packages

Choose a package by its purpose. Package names link to their documentation;
the Get started column links to quick starts or samples with setup instructions.

| Package | Purpose | Get started |
| --- | --- | --- |
| [langchain-azure-ai](./libs/azure-ai/README.md) | Microsoft Foundry models and embeddings, existing agents as graph nodes, LangGraph hosting, guardrails, tracing, tools, and Azure AI Search. | [Chat model quick start](#microsoft-foundry-models) · [Agent node notebook](./libs/azure-ai/docs/azure_ai_agents_demo_v2.ipynb) |
| [langchain-azure-compute](./libs/azure-compute/README.md) | Run agent-generated code and provide Deep Agents compute backends with Azure Container Apps dynamic sessions and sandboxes. | [Dynamic sessions quick start](./libs/azure-compute/README.md#dynamic-sessions) · [Sandbox quick start](./libs/azure-compute/README.md#sandboxes-deep-agents-backend) |
| [langchain-azure-cosmosdb](./libs/azure-cosmosdb/README.md) | Vector search, semantic caching, and chat history; LangGraph checkpoints, node caching, and long-term memory in Azure Cosmos DB. | [Cosmos DB samples](./samples/cosmosdb-nosql/README.md) |
| [langchain-sqlserver](./libs/sqlserver/README.md) | Vector search and chat history in Azure SQL and SQL Server. | [Azure SQL samples](https://github.com/Azure-Samples/azure-sql-langchain) |
| [langchain-azure-postgresql](./libs/azure-postgresql/README.md) | Vector search in Azure Database for PostgreSQL with Microsoft Entra ID authentication and DiskANN indexing. | [Vector store quick start](./libs/azure-postgresql/README.md#usage) |
| [langchain-azure-storage](./libs/azure-storage/README.md) | Load documents from Azure Blob Storage and persist Deep Agents files in blob containers. | [Document loader quick start](./libs/azure-storage/README.md#azure-blob-storage-document-loader-usage) · [Deep Agents samples](./samples/deepagents-storage-backend/README.md) |
| [langchain-azure-dynamic-sessions](./libs/azure-dynamic-sessions/README.md) | **Deprecated.** Azure Container Apps dynamic sessions integrations, replaced by `langchain-azure-compute`. | [Migration instructions](./libs/azure-dynamic-sessions/README.md) |

## Integrations

Browse by the interface your application needs:

- **Chat models and embedding models**: [Microsoft Foundry models](./libs/azure-ai/docs/azure_ai_models_demo.ipynb).
- **Tools and middleware**: [Azure AI tools](./libs/azure-ai/README.md#microsoft-foundry-tools) and [Content Safety guardrails](./libs/azure-ai/README.md#unified-moderation-guardrails).
- **Document loaders**: [Content Understanding](./libs/azure-ai/docs/content_understanding_loader_demo.ipynb) and [Azure Blob Storage](./libs/azure-storage/README.md#azure-blob-storage-document-loader-usage).
- **Vector stores**: [Azure AI Search](./samples/rag-storage-document-loaders/README.md), [Azure Cosmos DB](./libs/azure-cosmosdb/README.md#vector-store), [Azure SQL and SQL Server](./libs/sqlserver/README.md), and [Azure Database for PostgreSQL](./libs/azure-postgresql/README.md#usage).
- **Checkpointers and stores**: [Foundry State Store checkpoints](./libs/azure-ai/README.md#durable-graph-state), [Cosmos DB checkpoints](./libs/azure-cosmosdb/README.md#langgraph-checkpointer), and [Cosmos DB long-term memory](./libs/azure-cosmosdb/README.md#langgraph-store-long-term-memory).
- **Sandboxes and filesystem backends**: [Azure Container Apps compute](./libs/azure-compute/README.md) and [Azure Blob Storage files](./libs/azure-storage/README.md#deep-agents-azure-blob-storage-backend-usage).
- **Callbacks and tracing**: [Azure Application Insights](./libs/azure-ai/README.md#auto-tracing-to-azure-application-insights).

## LangGraph on Azure

### Microsoft Foundry Agent Service

Use `AgentServiceFactory.get_agent_node` to reference an agent that already exists
in Microsoft Foundry and use it as a node in your LangGraph graph. The agent runs
in Foundry and can be built independently of LangChain or LangGraph, using the
Foundry portal or other SDKs. See the [Agent Service notebook](./libs/azure-ai/docs/azure_ai_agents_demo_v2.ipynb).

### Durable graph state and memory

Persist graph state with [FoundryCheckpointSaver backed by Foundry State Store](./libs/azure-ai/README.md#durable-graph-state)
in Foundry-hosted agents, or the [Azure Cosmos DB checkpointer](./libs/azure-cosmosdb/README.md#langgraph-checkpointer).
For memory shared across threads, use the [Azure Cosmos DB store](./libs/azure-cosmosdb/README.md#langgraph-store-long-term-memory).

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

- [SDK samples](./samples) cover database, memory, and storage integrations.
- [SDK notebooks](./libs/azure-ai/docs) cover Foundry models, Microsoft Foundry Agent Service, Content Safety, and Content Understanding.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.
