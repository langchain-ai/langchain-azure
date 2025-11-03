# LangGraph React Agent

A containerized LangGraph agent using **Azure AI Foundry** components (Azure OpenAI, Application Insights), designed for cloud deployment.

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your project details.
   ```

3. **Run locally:**
   ```bash
   langgraph dev
   ```
   Server runs at `http://localhost:2024`

## Docker Deployment

This guide explains how to containerize and deploy your LangGraph application using Docker.

### Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)

### Building the Docker Image

```bash
docker-compose up --build
```

### Push to Azure Container Registry

```bash
# Login to Azure and authenticate with ACR
REGISTRY_NAME="REGISTRY_NAME"

az acr login --name REGISTRY_NAME

# Build and tag the image
docker build -t langgraph-agent .
docker tag langgraph-agent REGISTRY_NAME.azurecr.io/langgraph-agent:latest

# Push to ACR
docker push REGISTRY_NAME.azurecr.io/langgraph-agent:latest
```

### Deploy to ACI

```bash
# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name langgraph-agent \
  --image YOUR_REGISTRY_NAME.azurecr.io/langgraph-agent:latest \
  --ports 2024
```

## Client Usage

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2024")

async for chunk in client.runs.stream(
    None,
    "agent",
    input={"messages": [{"role": "human", "content": "I'm the coolest!"}]},
    stream_mode="updates",
):
    print(chunk.data)
```
