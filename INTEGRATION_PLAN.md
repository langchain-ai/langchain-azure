# Azure AI Foundry Enterprise Integration Plan
## LangChain Agents + Azure AI Foundry Hybrid Architecture

**Version**: 3.0 (Phase 1 Enterprise Implementation)  
**Last Updated**: 2026-01-26  
**Status**: 🚀 **APPROVED - READY FOR IMPLEMENTATION**

---

## Executive Summary

### Approved Architecture: Hybrid Approach

Your **9 production LangChain agents** will continue working with their current Azure OpenAI setup while gaining Azure AI Foundry enterprise capabilities through a **thin proxy layer**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ENTERPRISE ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐    │
│   │  Copilot Studio │    │   Power Platform  │    │    Teams/M365     │    │
│   │  (Native Agent) │    │   (Connector)     │    │    (Bot)          │    │
│   └────────┬────────┘    └────────┬─────────┘    └─────────┬──────────┘    │
│            │                      │                        │               │
│            └──────────────────────┼────────────────────────┘               │
│                                   ▼                                        │
│            ┌──────────────────────────────────────────────┐               │
│            │        Azure AI Foundry Project              │               │
│            │  ┌─────────────────────────────────────────┐ │               │
│            │  │     AI Foundry Proxy Agent              │ │               │
│            │  │  • Enterprise governance                │ │               │
│            │  │  • Azure RBAC & managed identity        │ │               │
│            │  │  • OpenTelemetry → App Insights         │ │               │
│            │  │  • Built-in tools (optional)            │ │               │
│            │  └──────────────────┬──────────────────────┘ │               │
│            └─────────────────────┼────────────────────────┘               │
│                                  │ OpenAPI / REST                          │
│                                  ▼                                        │
│            ┌──────────────────────────────────────────────┐               │
│            │     Your LangChain Agents (FastAPI)          │               │
│            │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │               │
│            │  │Helpdesk │ │Research │ │ServiceNow│ ...    │               │
│            │  └────┬────┘ └────┬────┘ └────┬────┘        │               │
│            │       └───────────┼───────────┘             │               │
│            │                   ▼                          │               │
│            │          Azure OpenAI (Direct)               │               │
│            └──────────────────────────────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Benefits
| Feature | Current State | After Phase 1 |
|---------|--------------|---------------|
| **Existing Agents** | ✅ Working | ✅ Unchanged, still working |
| **Enterprise RBAC** | ❌ Custom only | ✅ Azure AI Foundry roles |
| **Managed Identity** | ❌ API keys | ✅ DefaultAzureCredential |
| **Observability** | ✅ LangSmith | ✅ LangSmith + Azure Monitor |
| **Copilot Studio** | ❌ Not available | ✅ Native integration |
| **Compliance** | ⚠️ Custom | ✅ SOC 2, ISO 27001, HIPAA |
| **Private Networking** | ⚠️ Custom VNet | ✅ Managed VNet with Private Link |

---

## Phase 1: Enterprise Azure AI Foundry Setup (Implementation Guide)

### Overview

Phase 1 establishes the Azure AI Foundry infrastructure with enterprise-grade security, governance, and observability. Your existing LangChain agents remain unchanged while gaining access to Azure AI Foundry capabilities.

```
Timeline: Week 1-2
Effort: 2-3 days for infrastructure + 1-2 days for integration
Risk: LOW (no changes to existing agents)
```

---

## 1. Infrastructure as Code: Azure AI Foundry Hub & Project

### 1.1 Architecture Components

| Component | Purpose | Enterprise Features |
|-----------|---------|---------------------|
| **AI Hub** | Central governance, shared resources | CMK, RBAC, audit logs, managed network |
| **AI Project** | Agent workload isolation | Project-level RBAC, dedicated resources |
| **App Insights** | Observability & tracing | OpenTelemetry, distributed tracing |
| **Key Vault** | Secrets management | CMK, access policies, audit |
| **Storage Account** | Agent data storage | Private endpoints, encryption |

### 1.2 Bicep Template: Enterprise AI Foundry Hub

Create `infra/main.bicep`:

```bicep
// main.bicep - Enterprise Azure AI Foundry Hub with Project
// Based on Azure Verified Modules (AVM)

targetScope = 'resourceGroup'

@description('Location for all resources')
param location string = resourceGroup().location

@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'prod'

@description('Base name for resources')
param baseName string = 'langchain'

@description('Enable managed network isolation')
param enableManagedNetwork bool = true

@description('Admin principal IDs for RBAC')
param adminPrincipalIds array = []

// Variables
var uniqueSuffix = uniqueString(resourceGroup().id)
var hubName = 'aihub-${baseName}-${environment}-${uniqueSuffix}'
var projectName = 'aiproj-${baseName}-agents-${environment}'
var keyVaultName = 'kv-${baseName}-${uniqueSuffix}'
var storageAccountName = 'st${baseName}${uniqueSuffix}'
var appInsightsName = 'appi-${baseName}-${environment}'
var logAnalyticsName = 'log-${baseName}-${environment}'

// ============================================================================
// Supporting Resources
// ============================================================================

// Log Analytics Workspace (for Application Insights)
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 90
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

// Application Insights (OpenTelemetry tracing destination)
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// Key Vault (for secrets, API keys)
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enablePurgeProtection: true
    networkAcls: {
      defaultAction: 'Deny'
      bypass: 'AzureServices'
    }
  }
}

// Storage Account (for agent data)
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
    networkAcls: {
      defaultAction: 'Deny'
      bypass: 'AzureServices'
    }
  }
}

// ============================================================================
// Azure AI Foundry Hub (Central Governance)
// ============================================================================

resource aiHub 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: hubName
  location: location
  kind: 'Hub'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'LangChain Agents Hub (${environment})'
    description: 'Central hub for LangChain agent governance and shared resources'
    
    // Associated resources
    keyVault: keyVault.id
    storageAccount: storageAccount.id
    applicationInsights: appInsights.id
    
    // Managed network isolation (Enterprise security)
    managedNetwork: enableManagedNetwork ? {
      isolationMode: 'AllowInternetOutbound' // Or 'AllowOnlyApprovedOutbound' for stricter
      outboundRules: {
        azure_openai: {
          type: 'PrivateEndpoint'
          destination: {
            serviceResourceId: '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.CognitiveServices/accounts/*'
            subresourceTarget: 'account'
          }
        }
      }
    } : null
    
    // Public network access control
    publicNetworkAccess: 'Disabled'
    
    // V1 legacy mode disabled
    v1LegacyMode: false
  }
  
  tags: {
    environment: environment
    purpose: 'langchain-agents'
    managedBy: 'bicep'
  }
}

// ============================================================================
// Azure AI Foundry Project (Agent Workload)
// ============================================================================

resource aiProject 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: projectName
  location: location
  kind: 'Project'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'LangChain Agents Project'
    description: 'Project for running LangChain agents via Azure AI Foundry'
    
    // Link to hub
    hubResourceId: aiHub.id
    
    // Inherit hub settings
    publicNetworkAccess: 'Disabled'
    v1LegacyMode: false
  }
  
  tags: {
    environment: environment
    purpose: 'langchain-agents'
    managedBy: 'bicep'
  }
}

// ============================================================================
// RBAC Role Assignments (Enterprise Governance)
// ============================================================================

// Azure AI User role for admins on Hub
resource hubRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for principalId in adminPrincipalIds: {
  scope: aiHub
  name: guid(aiHub.id, principalId, 'Azure AI User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '53ca0ad7-0e25-4e31-b8a4-7c1e5bf3d3e4') // Azure AI User
    principalId: principalId
    principalType: 'User'
  }
}]

// Azure AI Developer role for project
resource projectRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for principalId in adminPrincipalIds: {
  scope: aiProject
  name: guid(aiProject.id, principalId, 'Azure AI Developer')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '64702f94-c441-49e6-a78b-ef80e0188fee') // Azure AI Developer
    principalId: principalId
    principalType: 'User'
  }
}]

// ============================================================================
// Outputs
// ============================================================================

@description('AI Hub resource ID')
output hubResourceId string = aiHub.id

@description('AI Hub name')
output hubName string = aiHub.name

@description('AI Project resource ID')
output projectResourceId string = aiProject.id

@description('AI Project name')
output projectName string = aiProject.name

@description('AI Project endpoint for SDK')
output projectEndpoint string = 'https://${location}.api.azureml.ms/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.MachineLearningServices/workspaces/${projectName}'

@description('Application Insights connection string')
output appInsightsConnectionString string = appInsights.properties.ConnectionString

@description('Key Vault URI')
output keyVaultUri string = keyVault.properties.vaultUri

@description('Storage Account name')
output storageAccountName string = storageAccount.name
```

### 1.3 Parameter Files

Create `infra/parameters.prod.json`:

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "environment": { "value": "prod" },
    "baseName": { "value": "langchain" },
    "enableManagedNetwork": { "value": true },
    "adminPrincipalIds": {
      "value": [
        "00000000-0000-0000-0000-000000000000"
      ]
    }
  }
}
```

### 1.4 Deployment Commands

```powershell
# Login and set subscription
az login
az account set --subscription "<your-subscription-id>"

# Create resource group
az group create --name rg-langchain-agents-prod --location eastus2

# Deploy infrastructure
az deployment group create `
  --resource-group rg-langchain-agents-prod `
  --template-file infra/main.bicep `
  --parameters @infra/parameters.prod.json `
  --name langchain-ai-foundry-$(Get-Date -Format 'yyyyMMddHHmm')

# Get outputs
$outputs = az deployment group show `
  --resource-group rg-langchain-agents-prod `
  --name langchain-ai-foundry-* `
  --query properties.outputs -o json | ConvertFrom-Json

# Save to .env
@"
AZURE_AI_PROJECT_ENDPOINT=$($outputs.projectEndpoint.value)
APPLICATIONINSIGHTS_CONNECTION_STRING=$($outputs.appInsightsConnectionString.value)
"@ | Out-File -FilePath .env.azure -Encoding utf8
```

---

## 2. Enterprise Security Configuration

### 2.1 RBAC Roles Reference

| Role | Scope | Permissions | Use Case |
|------|-------|-------------|----------|
| **Azure AI User** | Hub/Project | Read, inference, use models | End users, applications |
| **Azure AI Developer** | Project | Create agents, deploy models | Developers |
| **Azure AI Contributor** | Hub/Project | Full control except RBAC | DevOps |
| **Azure AI Administrator** | Hub | Full control + RBAC | Platform admins |

### 2.2 Managed Identity for Agent Applications

Your FastAPI application should use Managed Identity instead of API keys:

```python
# app/agents/base/credentials.py
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
import os

def get_azure_credential():
    """Get appropriate Azure credential based on environment.
    
    Priority:
    1. User-assigned Managed Identity (if configured)
    2. System-assigned Managed Identity (in Azure)
    3. DefaultAzureCredential (local dev with az login)
    """
    user_assigned_client_id = os.getenv("AZURE_CLIENT_ID")
    
    if user_assigned_client_id:
        # User-assigned managed identity
        return ManagedIdentityCredential(client_id=user_assigned_client_id)
    
    if os.getenv("IDENTITY_ENDPOINT"):
        # System-assigned managed identity (Azure environment)
        return ManagedIdentityCredential()
    
    # Local development - uses az login credentials
    return DefaultAzureCredential()
```

### 2.3 Private Endpoint Configuration (Optional - High Security)

Add to `main.bicep` for full network isolation:

```bicep
// Private DNS Zones
resource privateDnsZoneAzureML 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink.api.azureml.ms'
  location: 'global'
}

resource privateDnsZoneNotebooks 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink.notebooks.azure.net'
  location: 'global'
}

// Private Endpoint for Hub
resource hubPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-11-01' = {
  name: 'pe-${hubName}'
  location: location
  properties: {
    subnet: { id: '<your-subnet-id>' }
    privateLinkServiceConnections: [
      {
        name: 'hub-connection'
        properties: {
          privateLinkServiceId: aiHub.id
          groupIds: ['amlworkspace']
        }
      }
    ]
  }
}
```

---

## 3. Observability: Azure Monitor + OpenTelemetry Tracing

### 3.1 Tracing Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Observability Stack                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────┐                     ┌───────────────────────────┐ │
│   │  LangChain      │──── OpenTelemetry ──▶│   Application Insights   │ │
│   │  Agent Code     │                      │   (Azure Monitor)         │ │
│   └─────────────────┘                      └───────────────────────────┘ │
│           │                                            │                 │
│           │ LangSmith                                  │ Azure Portal    │
│           ▼                                            ▼                 │
│   ┌─────────────────┐                     ┌───────────────────────────┐ │
│   │   LangSmith     │                     │   Log Analytics          │ │
│   │   (Optional)    │                     │   Workbooks, Alerts      │ │
│   └─────────────────┘                     └───────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Install Tracing Dependencies

```bash
# Add to pyproject.toml or requirements.txt
pip install azure-monitor-opentelemetry opentelemetry-instrumentation-langchain
```

### 3.3 Tracing Setup Code

Create `app/observability/tracing.py`:

```python
"""
Enterprise observability with OpenTelemetry and Azure Application Insights.
Based on Microsoft best practices for Azure AI Foundry tracing.
"""

import os
from typing import Optional
import logging

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)


def configure_tracing(
    service_name: str = "langchain-agents",
    enable_langsmith: bool = True,
) -> None:
    """Configure enterprise tracing with Azure Monitor and optional LangSmith.
    
    Args:
        service_name: Name of the service for tracing
        enable_langsmith: Whether to also enable LangSmith tracing
    
    Environment Variables Required:
        APPLICATIONINSIGHTS_CONNECTION_STRING: Azure App Insights connection
        LANGCHAIN_TRACING_V2: "true" to enable LangSmith (optional)
        LANGCHAIN_API_KEY: LangSmith API key (optional)
    """
    
    # Get App Insights connection string
    app_insights_conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    
    if app_insights_conn:
        logger.info("Configuring Azure Monitor OpenTelemetry...")
        
        # Configure Azure Monitor with OpenTelemetry
        configure_azure_monitor(
            connection_string=app_insights_conn,
            service_name=service_name,
            enable_live_metrics=True,  # Real-time metrics
        )
        
        # Instrument LangChain for automatic tracing
        LangchainInstrumentor().instrument()
        
        logger.info("Azure Monitor tracing configured successfully")
    else:
        logger.warning("APPLICATIONINSIGHTS_CONNECTION_STRING not set - Azure Monitor tracing disabled")
    
    # LangSmith tracing (complementary, not replacement)
    if enable_langsmith:
        langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        if langsmith_enabled:
            logger.info("LangSmith tracing enabled via LANGCHAIN_TRACING_V2")
        else:
            logger.info("LangSmith tracing not enabled")


def get_tracer(name: str = "langchain-agents") -> trace.Tracer:
    """Get OpenTelemetry tracer for custom spans."""
    return trace.get_tracer(name)


# Context manager for custom traced operations
class TracedOperation:
    """Context manager for creating custom traced spans."""
    
    def __init__(self, name: str, attributes: Optional[dict] = None):
        self.name = name
        self.attributes = attributes or {}
        self.tracer = get_tracer()
        self.span = None
    
    def __enter__(self):
        self.span = self.tracer.start_span(self.name)
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
            self.span.record_exception(exc_val)
        self.span.end()
        return False
```

### 3.4 Initialize Tracing in FastAPI

Update `app/server.py`:

```python
from fastapi import FastAPI
from app.observability.tracing import configure_tracing

# Configure tracing BEFORE creating app
configure_tracing(service_name="langchain-agents")

app = FastAPI(title="LangChain Agents API")

@app.on_event("startup")
async def startup():
    """Application startup - tracing already configured."""
    pass
```

---

## 4. Agent Integration Layer (Hybrid Wrapper)

### 4.1 AI Foundry Proxy Agent

Create `app/agents/foundry/proxy_agent.py`:

```python
"""
Azure AI Foundry Proxy Agent - Thin wrapper that routes requests 
to your existing LangChain agents while providing AI Foundry integration.

This is the key to the hybrid approach:
- AI Foundry handles: Enterprise governance, RBAC, tracing, Copilot Studio
- Your agents handle: Actual business logic, custom tools, Azure OpenAI calls
"""

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    OpenApiTool,
    OpenApiAnonymousAuthDetails,
)
from azure.identity import DefaultAzureCredential
import os
from typing import Optional

from app.agents.base.credentials import get_azure_credential


class FoundryProxyAgent:
    """
    Azure AI Foundry agent that proxies to your LangChain FastAPI endpoints.
    
    Architecture:
        Copilot Studio → AI Foundry Agent → [OpenAPI Tool] → Your FastAPI → LangChain Agent
    
    Benefits:
        - Your agents unchanged (still use Azure OpenAI direct)
        - AI Foundry provides enterprise layer (RBAC, tracing, governance)
        - Copilot Studio can call your agents natively
    """
    
    def __init__(
        self,
        agent_name: str,
        fastapi_base_url: str,
        openapi_spec_url: Optional[str] = None,
    ):
        """
        Initialize the Foundry proxy agent.
        
        Args:
            agent_name: Name for the AI Foundry agent
            fastapi_base_url: Base URL of your LangChain FastAPI server
            openapi_spec_url: URL to OpenAPI spec (default: {base_url}/openapi.json)
        """
        self.agent_name = agent_name
        self.fastapi_base_url = fastapi_base_url.rstrip('/')
        self.openapi_spec_url = openapi_spec_url or f"{self.fastapi_base_url}/openapi.json"
        
        # Initialize AI Project client
        self.project_endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
        self.credential = get_azure_credential()
        self.project_client = AIProjectClient(
            endpoint=self.project_endpoint,
            credential=self.credential,
        )
        
        self._agent = None
        self._thread = None
    
    def create_agent(self, instructions: str) -> str:
        """
        Create an AI Foundry agent with OpenAPI tool pointing to your FastAPI.
        
        Args:
            instructions: System instructions for the agent
            
        Returns:
            agent_id: The created agent's ID
        """
        # Create OpenAPI tool that calls your LangChain agent endpoint
        openapi_tool = OpenApiTool(
            name=f"{self.agent_name}_api",
            description=f"API to interact with {self.agent_name}",
            spec={"url": self.openapi_spec_url},
            auth=OpenApiAnonymousAuthDetails(),  # Or configure auth as needed
        )
        
        # Create the AI Foundry agent
        self._agent = self.project_client.agents.create_agent(
            model="gpt-4.1",  # Model for routing/orchestration
            name=self.agent_name,
            instructions=instructions,
            tools=[openapi_tool.definitions],
        )
        
        return self._agent.id
    
    def create_thread(self) -> str:
        """Create a conversation thread."""
        self._thread = self.project_client.agents.create_thread()
        return self._thread.id
    
    def chat(self, message: str) -> str:
        """
        Send a message and get a response.
        The AI Foundry agent will route to your LangChain agent via OpenAPI.
        """
        if not self._agent or not self._thread:
            raise RuntimeError("Agent or thread not created. Call create_agent() and create_thread() first.")
        
        # Add user message to thread
        self.project_client.agents.create_message(
            thread_id=self._thread.id,
            role="user",
            content=message,
        )
        
        # Run the agent
        run = self.project_client.agents.create_and_process_run(
            thread_id=self._thread.id,
            agent_id=self._agent.id,
        )
        
        # Get the response
        messages = self.project_client.agents.list_messages(thread_id=self._thread.id)
        
        # Return the last assistant message
        for msg in messages.data:
            if msg.role == "assistant":
                return msg.content[0].text.value
        
        return "No response generated"
    
    def cleanup(self):
        """Clean up resources."""
        if self._agent:
            self.project_client.agents.delete_agent(self._agent.id)
        if self._thread:
            self.project_client.agents.delete_thread(self._thread.id)


# Example usage factory
def create_helpdesk_proxy() -> FoundryProxyAgent:
    """Create a proxy agent for IT Helpdesk."""
    return FoundryProxyAgent(
        agent_name="it-helpdesk-proxy",
        fastapi_base_url=os.getenv("LANGCHAIN_FASTAPI_URL", "http://localhost:8000"),
    )
```

### 4.2 Feature Flag Configuration

Update `app/config.py`:

```python
"""
Configuration with feature flags for hybrid Azure AI Foundry integration.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AzureConfig:
    """Azure AI configuration with feature flags."""
    
    # Feature Flags
    use_azure_foundry: bool = False  # Master switch for AI Foundry
    use_foundry_tracing: bool = True  # Use AI Foundry OpenTelemetry
    use_langsmith_tracing: bool = True  # Keep LangSmith as well
    
    # Azure AI Foundry
    project_endpoint: Optional[str] = None
    
    # Azure OpenAI (current/existing)
    openai_endpoint: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_deployment: str = "gpt-4o-mini"
    openai_api_version: str = "2024-08-01-preview"
    
    # Observability
    app_insights_connection: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "AzureConfig":
        """Load configuration from environment variables."""
        return cls(
            # Feature flags
            use_azure_foundry=os.getenv("USE_AZURE_FOUNDRY", "false").lower() == "true",
            use_foundry_tracing=os.getenv("USE_FOUNDRY_TRACING", "true").lower() == "true",
            use_langsmith_tracing=os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true",
            
            # Azure AI Foundry
            project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
            
            # Azure OpenAI
            openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
            openai_api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
            
            # Observability
            app_insights_connection=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
        )
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if self.use_azure_foundry and not self.project_endpoint:
            issues.append("USE_AZURE_FOUNDRY=true but AZURE_AI_PROJECT_ENDPOINT not set")
        
        if not self.openai_endpoint and not self.project_endpoint:
            issues.append("Neither AZURE_OPENAI_ENDPOINT nor AZURE_AI_PROJECT_ENDPOINT set")
        
        if self.use_foundry_tracing and not self.app_insights_connection:
            issues.append("USE_FOUNDRY_TRACING=true but APPLICATIONINSIGHTS_CONNECTION_STRING not set")
        
        return issues


# Global config instance
config = AzureConfig.from_env()
```

---

## 5. Environment Configuration

### 5.1 Complete `.env` Template

Create `.env.template`:

```bash
# =============================================================================
# Azure AI Foundry Enterprise Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Feature Flags (Control hybrid behavior)
# -----------------------------------------------------------------------------
USE_AZURE_FOUNDRY=false           # Set true to enable AI Foundry layer
USE_FOUNDRY_TRACING=true          # OpenTelemetry to App Insights
LANGCHAIN_TRACING_V2=true         # Keep LangSmith tracing

# -----------------------------------------------------------------------------
# Azure AI Foundry (Enterprise Layer)
# -----------------------------------------------------------------------------
AZURE_AI_PROJECT_ENDPOINT=https://<region>.api.azureml.ms/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<project>

# Managed Identity (preferred over API keys)
# AZURE_CLIENT_ID=<user-assigned-managed-identity-client-id>  # Optional

# -----------------------------------------------------------------------------
# Azure OpenAI (Current Production - DO NOT CHANGE)
# -----------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
OPENAI_API_VERSION=2024-08-01-preview

# For o4-mini models (special constraints)
# OPENAI_API_VERSION=2024-12-01-preview
# Note: o4-mini requires temperature=1.0

# -----------------------------------------------------------------------------
# Observability (Enterprise Monitoring)
# -----------------------------------------------------------------------------
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=<key>;IngestionEndpoint=https://<region>.in.applicationinsights.azure.com/

# LangSmith (Optional - runs alongside App Insights)
LANGCHAIN_API_KEY=<your-langsmith-key>
LANGCHAIN_PROJECT=langchain-agents-prod

# -----------------------------------------------------------------------------
# Agent Server Configuration
# -----------------------------------------------------------------------------
LANGCHAIN_FASTAPI_URL=http://localhost:8000
LANGCHAIN_FASTAPI_PORT=8000
```

### 5.2 Docker Configuration

Update `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Configure for Azure (Managed Identity support)
ENV AZURE_IDENTITY_ENABLE_LEGACY_TENANT_SELECTION=true

# Expose port
EXPOSE 8000

# Run with tracing enabled
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6. Copilot Studio Integration

### 6.1 Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Copilot Studio                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                          Custom Copilot                                 ││
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐          ││
│  │  │  Knowledge    │    │   Actions     │    │   Plugins     │          ││
│  │  │  (SharePoint) │    │ (Power Auto)  │    │ (AI Foundry)  │ ◄──┐     ││
│  │  └───────────────┘    └───────────────┘    └───────────────┘    │     ││
│  └─────────────────────────────────────────────────────────────────┼─────┘│
└────────────────────────────────────────────────────────────────────┼──────┘
                                                                     │
                                                    Azure AI Foundry Agent
                                                    (Proxy to your agents)
                                                                     │
                                                    ┌────────────────┴──────┐
                                                    │  Your LangChain       │
                                                    │  FastAPI Server       │
                                                    └───────────────────────┘
```

### 6.2 Steps to Connect Copilot Studio

1. **Create Azure AI Foundry Agent** (using proxy pattern from Section 4)

2. **Register in Copilot Studio**:
   - Go to https://copilotstudio.microsoft.com
   - Create new Custom Copilot or edit existing
   - Navigate to **Actions** → **Add action** → **Azure AI Foundry Agent**
   - Select your AI Foundry project and agent

3. **Configure Trigger Topics**:
   - Create topics that route to your agents
   - Example: "IT Support" topic → IT Helpdesk Agent

4. **Test in Copilot Studio**:
   - Use Test pane to verify routing
   - Check that responses come from your LangChain agents

---

## 7. Validation Checklist

### 7.1 Infrastructure Validation

```powershell
# Verify Hub created
az ml workspace show --name aihub-langchain-prod-* --resource-group rg-langchain-agents-prod

# Verify Project created
az ml workspace show --name aiproj-langchain-agents-prod --resource-group rg-langchain-agents-prod

# Test SDK connection
python -c "
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
import os

client = AIProjectClient(
    endpoint=os.environ['AZURE_AI_PROJECT_ENDPOINT'],
    credential=DefaultAzureCredential()
)
print('✅ Connected to Azure AI Foundry')
print(f'   Project: {client._config.endpoint}')
"
```

### 7.2 Tracing Validation

```python
# Test tracing is working
from app.observability.tracing import configure_tracing, TracedOperation

configure_tracing()

with TracedOperation("test-span", {"test": "value"}):
    print("✅ Traced operation executed")
    
# Check Azure Portal → Application Insights → Transaction Search
# Should see "test-span" within a few minutes
```

### 7.3 Agent Integration Validation

```python
# Test existing agents still work
from app.agents.it_helpdesk import ITHelpdeskAgent

agent = ITHelpdeskAgent()
response = agent.chat("Hello, I need help with my laptop", thread_id="test-1")
print(f"✅ Existing agent working: {response[:100]}...")

# Test AI Foundry proxy (if enabled)
import os
if os.getenv("USE_AZURE_FOUNDRY", "").lower() == "true":
    from app.agents.foundry.proxy_agent import create_helpdesk_proxy
    
    proxy = create_helpdesk_proxy()
    proxy.create_agent("You are an IT Helpdesk assistant")
    proxy.create_thread()
    response = proxy.chat("Hello, I need help")
    print(f"✅ AI Foundry proxy working: {response[:100]}...")
    proxy.cleanup()
```

---

## 8. Production Deployment

### 8.1 Azure Container Apps Deployment

```bicep
// container-app.bicep
resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'langchain-agents'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
      }
      secrets: [
        { name: 'azure-openai-key', value: azureOpenAIKey }
        { name: 'app-insights-conn', value: appInsightsConnectionString }
      ]
    }
    template: {
      containers: [
        {
          name: 'langchain-agents'
          image: '${acrName}.azurecr.io/langchain-agents:latest'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            { name: 'AZURE_OPENAI_ENDPOINT', value: azureOpenAIEndpoint }
            { name: 'AZURE_OPENAI_API_KEY', secretRef: 'azure-openai-key' }
            { name: 'AZURE_AI_PROJECT_ENDPOINT', value: aiProjectEndpoint }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'app-insights-conn' }
            { name: 'USE_AZURE_FOUNDRY', value: 'true' }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 10
      }
    }
  }
}

// Grant Managed Identity access to AI Foundry
resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: aiProject
  name: guid(containerApp.id, 'Azure AI User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '53ca0ad7-0e25-4e31-b8a4-7c1e5bf3d3e4')
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}
```

### 8.2 CI/CD with GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy LangChain Agents

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
  RESOURCE_GROUP: rg-langchain-agents-prod
  ACR_NAME: acrlangchainagents

jobs:
  deploy-infra:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Azure Login
        uses: azure/login@v2
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Deploy Bicep
        uses: azure/arm-deploy@v2
        with:
          subscriptionId: ${{ env.AZURE_SUBSCRIPTION_ID }}
          resourceGroupName: ${{ env.RESOURCE_GROUP }}
          template: ./infra/main.bicep
          parameters: ./infra/parameters.prod.json

  build-and-push:
    runs-on: ubuntu-latest
    needs: deploy-infra
    steps:
      - uses: actions/checkout@v4
      
      - name: Azure Login
        uses: azure/login@v2
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Build and Push to ACR
        run: |
          az acr build \
            --registry ${{ env.ACR_NAME }} \
            --image langchain-agents:${{ github.sha }} \
            --image langchain-agents:latest \
            .

  deploy-app:
    runs-on: ubuntu-latest
    needs: build-and-push
    steps:
      - uses: actions/checkout@v4
      
      - name: Azure Login
        uses: azure/login@v2
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Deploy Container App
        run: |
          az containerapp update \
            --name langchain-agents \
            --resource-group ${{ env.RESOURCE_GROUP }} \
            --image ${{ env.ACR_NAME }}.azurecr.io/langchain-agents:${{ github.sha }}
```

---

## 9. Summary: Phase 1 Implementation Checklist

### ✅ Pre-Implementation

- [ ] Review and approve this integration plan
- [ ] Verify Azure subscription access with Owner/Contributor role
- [ ] Confirm resource naming conventions and region selection
- [ ] Identify admin principal IDs for RBAC assignment

### ✅ Infrastructure Deployment (Day 1)

- [ ] Create resource group: `rg-langchain-agents-prod`
- [ ] Deploy Bicep template: `az deployment group create ...`
- [ ] Verify Hub and Project created in Azure Portal
- [ ] Note down Project Endpoint from outputs
- [ ] Note down App Insights Connection String from outputs

### ✅ Application Configuration (Day 1-2)

- [ ] Update `.env` with `AZURE_AI_PROJECT_ENDPOINT`
- [ ] Update `.env` with `APPLICATIONINSIGHTS_CONNECTION_STRING`
- [ ] Set `USE_AZURE_FOUNDRY=false` initially (start disabled)
- [ ] Install dependencies: `pip install azure-ai-projects azure-monitor-opentelemetry`

### ✅ Tracing Setup (Day 2)

- [ ] Create `app/observability/tracing.py`
- [ ] Update `app/server.py` to call `configure_tracing()`
- [ ] Verify traces appear in Application Insights (Transaction Search)
- [ ] Confirm LangSmith tracing still works (dual tracing)

### ✅ Agent Integration (Day 2-3)

- [ ] Create `app/agents/foundry/proxy_agent.py`
- [ ] Create `app/config.py` with feature flags
- [ ] Test existing agents still work (no changes needed)
- [ ] Test AI Foundry proxy with one agent (IT Helpdesk)

### ✅ Validation (Day 3)

- [ ] Run infrastructure validation script
- [ ] Run tracing validation script  
- [ ] Run agent integration validation script
- [ ] Verify Copilot Studio can discover agents (optional)

### ✅ Production Deployment (Day 3-4)

- [ ] Build and push Docker image to ACR
- [ ] Deploy Container App with managed identity
- [ ] Grant Container App identity access to AI Foundry Project
- [ ] Verify production deployment works
- [ ] Monitor Application Insights for production traces

---

## 10. Next Steps After Phase 1

### Phase 2: Enhanced Agent Capabilities (Week 3-4)

- Enable AI Foundry built-in tools for specific agents
- Implement Code Interpreter for Data Analyst Agent
- Implement File Search for Document Intelligence Agent
- A/B test AI Foundry agents vs existing agents

### Phase 3: Copilot Studio Integration (Week 5-6)

- Create Copilot Studio custom copilot
- Register AI Foundry agents as plugins
- Configure conversation topics and routing
- Deploy to Teams/M365 channels

### Phase 4: Advanced Enterprise Features (Week 7-8)

- Enable CMK encryption for compliance
- Configure private endpoints for high-security environments
- Implement custom dashboards in Azure Monitor
- Set up alerting and anomaly detection

---

## Appendix A: Reference Architecture

### A.1 Production Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              Azure Subscription                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                        Resource Group: rg-langchain-agents-prod                  │   │
│  ├─────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                  │   │
│  │  ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐     │   │
│  │  │  AI Hub           │     │  AI Project       │     │  Azure OpenAI     │     │   │
│  │  │  (Governance)     │────▶│  (Agents)         │────▶│  (Models)         │     │   │
│  │  │                   │     │                   │     │                   │     │   │
│  │  │  • RBAC policies  │     │  • Proxy agents   │     │  • gpt-4o-mini    │     │   │
│  │  │  • Shared config  │     │  • Tracing        │     │  • gpt-4.1        │     │   │
│  │  │  • Audit logs     │     │  • Secrets        │     │  • embeddings     │     │   │
│  │  └───────────────────┘     └─────────┬─────────┘     └───────────────────┘     │   │
│  │                                      │                                          │   │
│  │                                      │ OpenAPI                                  │   │
│  │                                      ▼                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────────┐     │   │
│  │  │              Container App: langchain-agents                           │     │   │
│  │  │  ┌─────────────────────────────────────────────────────────────────┐  │     │   │
│  │  │  │                    FastAPI + LangServe                          │  │     │   │
│  │  │  │                                                                 │  │     │   │
│  │  │  │  /agents/helpdesk  /agents/servicenow  /agents/research  ...    │  │     │   │
│  │  │  │       │                    │                  │                 │  │     │   │
│  │  │  │       └────────────────────┼──────────────────┘                 │  │     │   │
│  │  │  │                            ▼                                    │  │     │   │
│  │  │  │               LangGraph + LangChain Agents                      │  │     │   │
│  │  │  │                            │                                    │  │     │   │
│  │  │  │                            ▼                                    │  │     │   │
│  │  │  │                 AzureChatOpenAI (Direct)                        │  │     │   │
│  │  │  └─────────────────────────────────────────────────────────────────┘  │     │   │
│  │  └───────────────────────────────────────────────────────────────────────┘     │   │
│  │                                                                                  │   │
│  │  ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐     │   │
│  │  │  Application      │     │  Key Vault        │     │  Storage Account  │     │   │
│  │  │  Insights         │     │  (Secrets)        │     │  (Agent Data)     │     │   │
│  │  │  (Tracing)        │     │                   │     │                   │     │   │
│  │  └───────────────────┘     └───────────────────┘     └───────────────────┘     │   │
│  │                                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### A.2 RBAC Role Definitions

| Role | GUID | Permissions |
|------|------|-------------|
| Azure AI User | `53ca0ad7-0e25-4e31-b8a4-7c1e5bf3d3e4` | Read, inference, basic operations |
| Azure AI Developer | `64702f94-c441-49e6-a78b-ef80e0188fee` | Create agents, deploy models, manage connections |
| Azure AI Contributor | `5e0c9f58-4d4a-4d1d-9a6d-7f6e4d8b2a5c` | Full control except RBAC |
| Azure AI Administrator | `b59867f0-fa02-499b-be73-45a86b5b3e1c` | Full control including RBAC |

### A.3 Model Availability

| Model | Azure OpenAI | AI Foundry Agent | Notes |
|-------|-------------|-----------------|-------|
| gpt-4o | ✅ | ✅ | Recommended for agents |
| gpt-4o-mini | ✅ | ✅ | Cost-effective |
| gpt-4.1 | ✅ | ✅ | Latest reasoning |
| o4-mini | ✅ | ⚠️ | Requires temp=1.0, API 2024-12-01-preview |
| text-embedding-3 | ✅ | ✅ | For RAG |

---

## Appendix B: Troubleshooting

### B.1 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Authentication failed | `DefaultAzureCredential` error | Run `az login`, verify subscription access |
| Project not found | 404 error on endpoint | Verify `AZURE_AI_PROJECT_ENDPOINT` format |
| Tracing not appearing | No data in App Insights | Check `APPLICATIONINSIGHTS_CONNECTION_STRING`, wait 2-3 min |
| Model not available | Model deployment error | Verify model deployed in Azure OpenAI resource |
| RBAC denied | 403 Forbidden | Assign Azure AI User role to identity |

### B.2 Validation Scripts

```python
# validate_setup.py
import os
import sys

def validate():
    errors = []
    warnings = []
    
    # Check required environment variables
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
    ]
    
    optional = [
        "AZURE_AI_PROJECT_ENDPOINT",
        "APPLICATIONINSIGHTS_CONNECTION_STRING",
    ]
    
    for var in required:
        if not os.getenv(var):
            errors.append(f"Missing required: {var}")
    
    for var in optional:
        if not os.getenv(var):
            warnings.append(f"Missing optional: {var}")
    
    # Print results
    print("=" * 50)
    print("Configuration Validation")
    print("=" * 50)
    
    if errors:
        print("\n❌ ERRORS:")
        for e in errors:
            print(f"   - {e}")
    
    if warnings:
        print("\n⚠️ WARNINGS:")
        for w in warnings:
            print(f"   - {w}")
    
    if not errors:
        print("\n✅ Configuration valid!")
        
        # Test connections
        print("\nTesting connections...")
        
        # Test Azure OpenAI
        try:
            from langchain_openai import AzureChatOpenAI
            model = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
            )
            response = model.invoke("Hello")
            print("   ✅ Azure OpenAI connected")
        except Exception as e:
            print(f"   ❌ Azure OpenAI failed: {e}")
        
        # Test AI Foundry (if configured)
        if os.getenv("AZURE_AI_PROJECT_ENDPOINT"):
            try:
                from azure.ai.projects import AIProjectClient
                from azure.identity import DefaultAzureCredential
                client = AIProjectClient(
                    endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
                    credential=DefaultAzureCredential(),
                )
                print("   ✅ Azure AI Foundry connected")
            except Exception as e:
                print(f"   ❌ Azure AI Foundry failed: {e}")
    
    return len(errors) == 0

if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
```

---

## Appendix C: Cost Estimation

### C.1 Monthly Cost Breakdown (Production)

| Resource | SKU | Estimated Cost/Month |
|----------|-----|---------------------|
| AI Hub | N/A (free) | $0 |
| AI Project | N/A (free) | $0 |
| Azure OpenAI | Pay-per-use | ~$50-500 (usage based) |
| Container App | 1 vCPU, 2GB | ~$50 |
| Application Insights | 5GB/day | ~$15 |
| Key Vault | Standard | ~$1 |
| Storage Account | Standard LRS | ~$5 |
| **Total** | | **~$120-570/month** |

### C.2 Cost Optimization Tips

1. **Use gpt-4o-mini** instead of gpt-4o for most agents (10x cheaper)
2. **Enable auto-scaling** for Container App (scale to zero in dev)
3. **Set retention limits** on Application Insights (30 days vs 90 days)
4. **Use Managed Identity** (no API key rotation overhead)

---

**Document Version**: 3.0 (Phase 1 Enterprise Implementation)  
**Last Updated**: 2026-01-26  
**Status**: 🚀 **APPROVED - READY FOR IMPLEMENTATION**  
**Author**: GitHub Copilot (Agent Mode)
