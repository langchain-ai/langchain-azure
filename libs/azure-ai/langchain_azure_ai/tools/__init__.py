"""Tools provided by Azure AI Foundry."""

import importlib
from typing import TYPE_CHECKING, Any, List

from langchain_core.tools.base import BaseTool, BaseToolkit

from langchain_azure_ai._resources import AIServicesService

if TYPE_CHECKING:
    from langchain_azure_ai.tools.image_gen import OpenAIModelImageGenTool
    from langchain_azure_ai.tools.logic_apps import AzureLogicAppTool
    from langchain_azure_ai.tools.services.document_intelligence import (
        AzureAIDocumentIntelligenceTool,
    )
    from langchain_azure_ai.tools.services.image_analysis import (
        AzureAIImageAnalysisTool,
    )
    from langchain_azure_ai.tools.services.text_analytics_health import (
        AzureAITextAnalyticsHealthTool,
    )

# Mapping of lazy-loaded symbol names to their module paths
_MODULE_MAP = {
    "AzureAIDocumentIntelligenceTool": (
        "langchain_azure_ai.tools.services.document_intelligence"
    ),
    "AzureAIImageAnalysisTool": "langchain_azure_ai.tools.services.image_analysis",
    "AzureAITextAnalyticsHealthTool": (
        "langchain_azure_ai.tools.services.text_analytics_health"
    ),
    "OpenAIModelImageGenTool": "langchain_azure_ai.tools.image_gen",
    "AzureLogicAppTool": "langchain_azure_ai.tools.logic_apps",
}


def __getattr__(name: str) -> Any:
    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class AIServicesToolkit(BaseToolkit, AIServicesService):
    """Toolkit for Azure AI Services."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        from langchain_azure_ai.tools.services.document_intelligence import (
            AzureAIDocumentIntelligenceTool,
        )
        from langchain_azure_ai.tools.services.image_analysis import (
            AzureAIImageAnalysisTool,
        )
        from langchain_azure_ai.tools.services.text_analytics_health import (
            AzureAITextAnalyticsHealthTool,
        )

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
    "AIServicesToolkit",
    "AzureLogicAppTool",
    "OpenAIModelImageGenTool",
]
