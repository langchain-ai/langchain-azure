"""Tools provided by Azure AI Foundry."""

from typing import List

from langchain_core.tools.base import BaseTool, BaseToolkit

from langchain_azure_ai._resources import AIServicesService
from langchain_azure_ai.tools.ai_services.document_intelligence import (
    AzureAIDocumentIntelligenceTool,
)
from langchain_azure_ai.tools.ai_services.image_analysis import AzureAIImageAnalysisTool
from langchain_azure_ai.tools.ai_services.text_analytics_health import (
    AzureAITextAnalyticsHealthTool,
)


class AIServicesToolkit(BaseToolkit, AIServicesService):
    """Toolkit for Azure AI Services."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            AzureAIDocumentIntelligenceTool(
                endpoint=self.endpoint,
                credential=self.credential,
                api_version=self.api_version,
            ),
            AzureAIImageAnalysisTool(
                endpoint=self.endpoint,
                credential=self.credential,
                api_version=self.api_version,
            ),
            AzureAITextAnalyticsHealthTool(
                endpoint=self.endpoint,
                credential=self.credential,
                api_version=self.api_version,
            ),
        ]


__all__ = [
    "AzureAIDocumentIntelligenceTool",
    "AzureAIImageAnalysisTool",
    "AzureAITextAnalyticsHealthTool",
]
