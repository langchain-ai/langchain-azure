"""Tool that queries the Azure AI Text Analytics for Health API."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr, model_validator

from langchain_azure_ai._resources import AIServicesService

try:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    raise ImportError(
        "azure-ai-textanalytics is not installed. "
        "Run `pip install azure-ai-textanalytics` to install."
    )

logger = logging.getLogger(__name__)


class AzureAITextAnalyticsHealthTool(BaseTool, AIServicesService):
    """Tool that queries the Azure AI Text Analytics for Health API."""

    _client: TextAnalyticsClient = PrivateAttr()  # pyright: ignore[reportUndefinedVariable]

    name: str = "azure_ai_text_analytics_health"
    """The name of the tool."""

    description: str = (
        "A tool with access to Azure AI Services Text Analytics for Health. "
        "Useful for when you need to analyze healthcare-related text. "
        "Input should be a piece of text."
    )

    language: Optional[str] = None
    """The language of the input text. If not specified, the default language 
    configured in the Azure resource will be used."""

    country_hint: Optional[str] = None
    """The country hint for the input text. If not specified, the default country
    hint configured in the Azure resource will be used."""

    @model_validator(mode="after")
    def initialize_client(self) -> AzureAITextAnalyticsHealthTool:
        """Initialize the Azure AI Text Analytics client."""
        credential = (
            AzureKeyCredential(self.credential)
            if isinstance(self.credential, str)
            else self.credential
        )

        self._client = TextAnalyticsClient(
            endpoint=self.endpoint,  # type: ignore[arg-type]
            credential=credential,  # type: ignore[arg-type]
            default_language=self.language,
            default_country_hint=self.country_hint,
            **self.client_kwargs,
        )
        return self

    def _text_analysis(self, text: str) -> Dict:
        poller = self._client.begin_analyze_healthcare_entities(
            [{"id": "1", "text": text}]
        )

        result = poller.result()

        res_dict = {}

        docs = [doc for doc in result if not doc.is_error]

        if docs is not None:
            res_dict["entities"] = [
                f"{x.text} is a healthcare entity of type {x.category}"
                for y in docs
                for x in y.entities
            ]

        return res_dict

    def _format_text_analysis_result(self, text_analysis_result: Dict) -> str:
        formatted_result = []
        if "entities" in text_analysis_result:
            formatted_result.append(
                f"""The text contains the following healthcare entities: {
                    ", ".join(text_analysis_result["entities"])
                }""".replace("\n", " ")
            )

        return "\n".join(formatted_result)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            text_analysis_result = self._text_analysis(query)

            return self._format_text_analysis_result(text_analysis_result)
        except Exception as e:
            raise RuntimeError(f"Error while running {self.name}: {e}")
