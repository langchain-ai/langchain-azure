"""Tools provided by Azure AI Foundry."""

import importlib
from typing import TYPE_CHECKING, Any, List

from langchain_core.tools.base import BaseTool, BaseToolkit

from langchain_azure_ai._resources import AIServicesService

if TYPE_CHECKING:
    from langchain_azure_ai.tools._openai_tools import (
        AzureOpenAIModelImageGenTool,
        AzureOpenAITranscriptionsTool,
        ImageGenerationInput,
        SpeechToTextInput,
    )
    from langchain_azure_ai.tools.logic_apps import AzureLogicAppTool
    from langchain_azure_ai.tools.services.content_understanding import (
        AzureAIContentUnderstandingTool,
    )
    from langchain_azure_ai.tools.services.document_intelligence import (
        AzureAIDocumentIntelligenceTool,
    )
    from langchain_azure_ai.tools.services.image_analysis import (
        AzureAIImageAnalysisTool,
    )
    from langchain_azure_ai.tools.services.speech_to_text import (
        AzureAISpeechToTextTool,
    )
    from langchain_azure_ai.tools.services.text_analytics_health import (
        AzureAITextAnalyticsHealthTool,
    )
    from langchain_azure_ai.tools.services.text_to_speech import (
        AzureAITextToSpeechTool,
    )

# Mapping of lazy-loaded symbol names to their module paths
_MODULE_MAP = {
    "AzureAIContentUnderstandingTool": (
        "langchain_azure_ai.tools.services.content_understanding"
    ),
    "AzureAIDocumentIntelligenceTool": (
        "langchain_azure_ai.tools.services.document_intelligence"
    ),
    "AzureAIImageAnalysisTool": "langchain_azure_ai.tools.services.image_analysis",
    "AzureAISpeechToTextTool": ("langchain_azure_ai.tools.services.speech_to_text"),
    "AzureAITextToSpeechTool": ("langchain_azure_ai.tools.services.text_to_speech"),
    "AzureAITextAnalyticsHealthTool": (
        "langchain_azure_ai.tools.services.text_analytics_health"
    ),
    "AzureOpenAIModelImageGenTool": "langchain_azure_ai.tools._openai_tools",
    "AzureOpenAITranscriptionsTool": "langchain_azure_ai.tools._openai_tools",
    "ImageGenerationInput": "langchain_azure_ai.tools._openai_tools",
    "SpeechToTextInput": "langchain_azure_ai.tools._openai_tools",
    "AzureLogicAppTool": "langchain_azure_ai.tools.logic_apps",
}

# Re-export the builtin subpackage so ``from langchain_azure_ai.tools import builtin``
# works without an explicit import.
from langchain_azure_ai.tools import builtin as builtin  # noqa: E402


def __getattr__(name: str) -> Any:
    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class AIServicesToolkit(BaseToolkit, AIServicesService):
    """Toolkit for Azure AI Services."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        from langchain_azure_ai.tools.services.content_understanding import (
            AzureAIContentUnderstandingTool,
        )
        from langchain_azure_ai.tools.services.document_intelligence import (
            AzureAIDocumentIntelligenceTool,
        )
        from langchain_azure_ai.tools.services.image_analysis import (
            AzureAIImageAnalysisTool,
        )
        from langchain_azure_ai.tools.services.speech_to_text import (
            AzureAISpeechToTextTool,
        )
        from langchain_azure_ai.tools.services.text_analytics_health import (
            AzureAITextAnalyticsHealthTool,
        )
        from langchain_azure_ai.tools.services.text_to_speech import (
            AzureAITextToSpeechTool,
        )

        tools: List[BaseTool] = [
            AzureAIContentUnderstandingTool(
                endpoint=self.endpoint,
                credential=self.credential,
                api_version=self.api_version,
            ),
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
            AzureAISpeechToTextTool(
                endpoint=self.endpoint,
                credential=self.credential,
            ),
            AzureAITextToSpeechTool(
                endpoint=self.endpoint,
                credential=self.credential,
            ),
            AzureAITextAnalyticsHealthTool(
                endpoint=self.endpoint,
                credential=self.credential,
                api_version=self.api_version,
            ),
        ]
        return tools


__all__ = [
    "AzureAIContentUnderstandingTool",
    "AzureAIDocumentIntelligenceTool",
    "AzureAIImageAnalysisTool",
    "AzureAISpeechToTextTool",
    "AzureAITextToSpeechTool",
    "AzureAITextAnalyticsHealthTool",
    "AIServicesToolkit",
    "AzureLogicAppTool",
    "AzureOpenAIModelImageGenTool",
    "AzureOpenAITranscriptionsTool",
    "ImageGenerationInput",
    "SpeechToTextInput",
]
