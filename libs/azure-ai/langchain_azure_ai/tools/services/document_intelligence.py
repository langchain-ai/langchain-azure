"""Tool that queries the Azure AI Document Intelligence API."""

from __future__ import annotations

import base64
import logging
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, PrivateAttr, SkipValidation, model_validator

from langchain_azure_ai._resources import AIServicesService

try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential
except ImportError as ex:
    raise ImportError(
        "To use Azure AI Document Intelligence tool, please install the"
        "'azure-ai-documentintelligence' package: "
        "`pip install azure-ai-documentintelligence` or install the 'tools' "
        "extra: `pip install langchain-azure-ai[tools]`"
    ) from ex

logger = logging.getLogger(__name__)


class DocumentInput(BaseModel):
    """The input document for the Azure AI Document Intelligence tool."""

    source_type: Literal["url", "path", "base64"] = "url"
    """The type of the document source, either 'url', 'path', or 'base64'."""

    source: str
    """The document source, either a URL, a local file path, or a base64 string."""


class AzureAIDocumentIntelligenceTool(BaseTool, AIServicesService):
    """Tool that queries the Azure AI Document Intelligence API."""

    _client: DocumentIntelligenceClient = PrivateAttr()  # pyright: ignore[reportUndefinedVariable]

    name: str = "azure_ai_document_intelligence"
    """The name of the tool."""

    description: str = (
        "Extracts structured content from documents using Azure AI Document "
        "Intelligence. Analyzes PDFs, images, and Office files to extract text, "
        "tables, key-value pairs, and form fields with high-accuracy OCR. Ideal for "
        "parsing invoices, receipts, forms, contracts, and documents requiring "
        "structured data extraction. Accepts file paths, URLs, or base64 strings."
        "Returns text content, tables with preserved formatting, and document "
        "metadata."
    )
    """The description of the tool."""

    args_schema: Annotated[Optional[ArgsSchema], SkipValidation()] = DocumentInput
    """The input args schema for the tool."""

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

    def _document_analysis(self, source: str, source_type: str) -> Dict:
        """Analyze a document using the Document Intelligence client."""
        if source_type == "base64" or (
            source_type == "url" and source.startswith("data:")
        ):
            if source.startswith("data:"):
                base64_content = source.split(",", 1)[1]
            else:
                base64_content = source

            document_bytes = base64.b64decode(base64_content)
            poller = self._client.begin_analyze_document(
                model_id=self.model_id,
                body=AnalyzeDocumentRequest(bytes_source=document_bytes),  # type: ignore[call-overload]
            )
        elif source_type == "local":
            with open(source, "rb") as document:
                poller = self._client.begin_analyze_document(
                    model_id=self.model_id,
                    body=AnalyzeDocumentRequest(bytes_source=document),  # type: ignore[call-overload]
                )
        elif source_type == "url":
            poller = self._client.begin_analyze_document(
                model_id=self.model_id, body=AnalyzeDocumentRequest(url_source=source)
            )
        else:
            raise ValueError(f"Invalid document source type: {source_type}")

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
        source: str,
        source_type: str = "url",
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        # try:
        document_analysis_result = self._document_analysis(
            source, source_type=source_type
        )
        if not document_analysis_result:
            return "No good document analysis result was found"

        return self._format_document_analysis_result(document_analysis_result)
        # except Exception as ex:
        #    raise RuntimeError(f"Error while running {self.name}: {ex}") from ex
