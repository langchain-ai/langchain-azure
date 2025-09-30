"""Tool that queries the Azure AI Document Intelligence API."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr, model_validator

from langchain_azure_ai._resources import AIServicesService
from langchain_azure_ai.utils.utils import detect_file_src_type

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    raise ImportError(
        "azure-ai-documentintelligence is not installed. "
        "Run `pip install azure-ai-documentintelligence` to install."
    )

logger = logging.getLogger(__name__)


class AzureAIDocumentIntelligenceTool(BaseTool, AIServicesService):
    """Tool that queries the Azure AI Document Intelligence API."""

    _client: DocumentIntelligenceClient = PrivateAttr()  # pyright: ignore[reportUndefinedVariable]

    name: str = "azure_ai_document_intelligence"
    """The name of the tool."""

    description: str = (
        "A tool that uses Azure AI Document Intelligence API to analyze "
        "documents. Useful for when you need to extract text, tables, and "
        "key-value pairs from documents. Input should be a url or path to "
        "a document."
    )
    """The description of the tool."""

    model_id: str = "prebuilt-layout"
    """The model ID to use for document analysis. If not specified, the 
    prebuilt-document model will be used."""

    @model_validator(mode="after")
    def initialize_client(self) -> AzureAIDocumentIntelligenceTool:
        """Initialize the Azure AI Document Intelligence client."""
        credential = (
            AzureKeyCredential(self.credential)
            if isinstance(self.credential, str)
            else self.credential
        )

        self._client = DocumentIntelligenceClient(
            endpoint=self.endpoint,  # type: ignore[arg-type]
            credential=credential,  # type: ignore[arg-type]
            **self.client_kwargs,  # type: ignore[arg-type]
        )
        return self

    def _parse_tables(self, tables: List[Any]) -> List[Any]:
        """Parse tables from the document analysis result."""
        result = []
        for table in tables:
            rc, cc = table.row_count, table.column_count
            _table = [["" for _ in range(cc)] for _ in range(rc)]
            for cell in table.cells:
                _table[cell.row_index][cell.column_index] = cell.content
            result.append(_table)
        return result

    def _parse_kv_pairs(self, kv_pairs: List[Any]) -> List[Any]:
        """Parse key-value pairs from the document analysis result."""
        result = []
        for kv_pair in kv_pairs:
            key = kv_pair.key.content if kv_pair.key else ""
            value = kv_pair.value.content if kv_pair.value else ""
            result.append((key, value))
        return result

    def _document_analysis(self, document_path: str) -> Dict:
        """Analyze a document using the Document Intelligence client."""
        document_src_type = detect_file_src_type(document_path)
        if document_src_type == "local":
            with open(document_path, "rb") as document:
                poller = self._client.begin_analyze_document(
                    self.model_id,
                    AnalyzeDocumentRequest(bytes_source=document),  # type: ignore[call-overload]
                )
        elif document_src_type == "remote":
            poller = self._client.begin_analyze_document(
                self.model_id, AnalyzeDocumentRequest(url_source=document_path)
            )
        else:
            raise ValueError(f"Invalid document path: {document_path}")

        result = poller.result()
        res_dict = {}

        if result.content is not None:
            res_dict["content"] = result.content

        if result.tables is not None:
            res_dict["tables"] = self._parse_tables(result.tables)  # type: ignore[assignment]

        if result.key_value_pairs is not None:
            res_dict["key_value_pairs"] = self._parse_kv_pairs(result.key_value_pairs)  # type: ignore[assignment]

        return res_dict

    def _format_document_analysis_result(self, document_analysis_result: Dict) -> str:
        """Format the document analysis result into a readable string."""
        formatted_result = []
        if "content" in document_analysis_result:
            formatted_result.append(
                f"Content: {document_analysis_result['content']}".replace("\n", " ")
            )

        if "tables" in document_analysis_result:
            for i, table in enumerate(document_analysis_result["tables"]):
                formatted_result.append(f"Table {i}: {table}".replace("\n", " "))

        if "key_value_pairs" in document_analysis_result:
            for kv_pair in document_analysis_result["key_value_pairs"]:
                formatted_result.append(
                    f"{kv_pair[0]}: {kv_pair[1]}".replace("\n", " ")
                )

        return "\n".join(formatted_result)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            document_analysis_result = self._document_analysis(query)
            if not document_analysis_result:
                return "No good document analysis result was found"

            return self._format_document_analysis_result(document_analysis_result)
        except Exception as e:
            raise RuntimeError(f"Error while running {self.name}: {e}")
