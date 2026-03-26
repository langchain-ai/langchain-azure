"""Azure Content Understanding document loader for LangChain."""

from __future__ import annotations

import logging
import mimetypes
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.ai.contentunderstanding.models import (
    AnalysisInput,
    AnalysisResult,
)
from azure.core.credentials import AzureKeyCredential, TokenCredential
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_USER_AGENT = "langchain-azure-ai-cu-loader/1.0.0"

_DEFAULT_ANALYZERS: Dict[str, str] = {
    "application/": "prebuilt-documentSearch",
    "text/": "prebuilt-documentSearch",
    "image/": "prebuilt-documentSearch",
    "audio/": "prebuilt-audioSearch",
    "video/": "prebuilt-videoSearch",
}

_VALID_OUTPUT_MODES = ("markdown", "page", "segment")


class AzureContentUnderstandingLoader(BaseLoader):
    """Load documents, images, audio, and video using Azure Content Understanding.

    Produces LangChain Document objects with extracted markdown content
    and rich metadata (fields, confidence scores, source info).

    Exactly one of ``file_path``, ``url``, or ``bytes_source`` must be provided.

    Example:
        .. code-block:: python

            from azure.identity import DefaultAzureCredential
            from langchain_azure_ai.document_loaders import (
                AzureContentUnderstandingLoader,
            )

            loader = AzureContentUnderstandingLoader(
                endpoint="https://my-resource.services.ai.azure.com",
                credential=DefaultAzureCredential(),
                file_path="report.pdf",
            )
            docs = loader.load()
    """

    def __init__(
        self,
        endpoint: str,
        credential: Union[str, AzureKeyCredential, TokenCredential],
        *,
        analyzer_id: Optional[str] = None,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        bytes_source: Optional[bytes] = None,
        source: Optional[str] = None,
        output_mode: str = "markdown",
        content_range: Optional[str] = None,
        output_selection: Optional[List[str]] = None,
        model_deployments: Optional[Dict[str, str]] = None,
        analyze_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the loader.

        Args:
            endpoint: CU resource endpoint URL.
            credential: Azure credential — API key string,
                ``AzureKeyCredential``, or ``TokenCredential``.
            analyzer_id: Analyzer to use. Defaults by input MIME type if omitted.
            file_path: Path to a local file.
            url: URL to the content.
            bytes_source: Raw bytes of the content.
            source: Label for ``metadata["source"]``. Defaults to *file_path*
                or *url* when provided.
            output_mode: How to split results into Documents —
                ``"markdown"`` (default), ``"page"``, or ``"segment"``.
            content_range: Subset of input to analyze. Pages use ``"1-3,5,9-"``;
                audio/video uses milliseconds ``"0-60000"``.
            output_selection: What to include in metadata, e.g.
                ``["fields", "tables"]``. Defaults to include fields.
            model_deployments: Mapping of model names to deployment names.
                Required for custom analyzers that use ``generate`` or
                ``classify`` field methods.
            analyze_kwargs: Extra keyword arguments forwarded to
                ``begin_analyze`` (e.g., ``processing_location``).
        """
        if not endpoint or not str(endpoint).strip():
            raise ValueError("endpoint must be a non-empty string.")

        sources = [file_path, url, bytes_source]
        if sum(s is not None for s in sources) != 1:
            raise ValueError(
                "Exactly one of file_path, url, or bytes_source must be provided."
            )

        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUT_MODES}, "
                f"got '{output_mode}'"
            )

        if isinstance(credential, str):
            credential = AzureKeyCredential(credential)

        self._endpoint = endpoint
        self._credential = credential
        self._file_path = file_path
        self._url = url
        self._bytes_source = bytes_source
        self._output_mode = output_mode
        self._content_range = content_range
        self._output_selection = output_selection
        self._model_deployments = model_deployments
        self._analyze_kwargs = analyze_kwargs or {}

        # Resolve source label for metadata
        if source is not None:
            self._source = source
        elif file_path is not None:
            self._source = file_path
        elif url is not None:
            self._source = url
        else:
            self._source = "bytes_input"

        # Resolve MIME type and analyzer
        self._mime_type = self._detect_mime_type()
        self._analyzer_id = analyzer_id or self._resolve_default_analyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lazy_load(self) -> Iterator[Document]:
        """Load documents synchronously.

        Yields:
            ``Document`` objects parsed from the CU analysis result.
        """
        client = ContentUnderstandingClient(
            endpoint=self._endpoint,
            credential=self._credential,  # type: ignore[arg-type]
            user_agent=_USER_AGENT,
        )

        try:
            analysis_input = self._build_analysis_input()
            poller = client.begin_analyze(
                analyzer_id=self._analyzer_id,
                inputs=[analysis_input],
                model_deployments=self._model_deployments,
                **self._analyze_kwargs,
            )
            operation_id: Optional[str] = getattr(poller, "operation_id", None)
            if not isinstance(operation_id, str):
                operation_id = None
            result: AnalysisResult = poller.result()

            if isinstance(result.warnings, list):
                for warning in result.warnings:
                    logger.warning("CU analysis warning: %s", warning.message)

            if not result.contents:
                logger.warning("CU analysis returned no content items.")

            yield from self._map_result_to_documents(result, operation_id=operation_id)
        finally:
            client.close()

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Load documents asynchronously using CU's native async client.

        Yields:
            ``Document`` objects parsed from the CU analysis result.
        """
        from azure.ai.contentunderstanding.aio import (
            ContentUnderstandingClient as AsyncContentUnderstandingClient,
        )

        client = AsyncContentUnderstandingClient(
            endpoint=self._endpoint,
            credential=self._credential,  # type: ignore[arg-type]
            user_agent=_USER_AGENT,
        )
        try:
            analysis_input = self._build_analysis_input()
            poller = await client.begin_analyze(
                analyzer_id=self._analyzer_id,
                inputs=[analysis_input],
                model_deployments=self._model_deployments,
                **self._analyze_kwargs,
            )
            operation_id: Optional[str] = getattr(poller, "operation_id", None)
            if not isinstance(operation_id, str):
                operation_id = None
            result: AnalysisResult = await poller.result()

            if isinstance(result.warnings, list):
                for warning in result.warnings:
                    logger.warning("CU analysis warning: %s", warning.message)

            if not result.contents:
                logger.warning("CU analysis returned no content items.")

            for doc in self._map_result_to_documents(
                result, operation_id=operation_id
            ):
                yield doc
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    def _detect_mime_type(self) -> Optional[str]:
        """Detect MIME type from file path or URL."""
        path = self._file_path or self._url
        if path:
            mime_type, _ = mimetypes.guess_type(path)
            return mime_type
        return None

    def _resolve_default_analyzer(self) -> str:
        """Pick a default analyzer based on MIME type prefix."""
        if self._mime_type:
            for prefix, analyzer in _DEFAULT_ANALYZERS.items():
                if self._mime_type.startswith(prefix):
                    return analyzer
        return "prebuilt-documentSearch"

    def _build_analysis_input(self) -> AnalysisInput:
        """Build an ``AnalysisInput`` from the bound input source."""
        input_url: Optional[str] = None
        input_data: Optional[bytes] = None

        if self._url:
            input_url = self._url
        elif self._file_path:
            with open(self._file_path, "rb") as f:
                input_data = f.read()
        else:
            input_data = self._bytes_source

        return AnalysisInput(
            url=input_url,
            data=input_data,
            content_range=self._content_range,
        )

    # ------------------------------------------------------------------
    # Result → Document mapping
    # ------------------------------------------------------------------

    def _map_result_to_documents(
        self,
        result: AnalysisResult,
        *,
        operation_id: Optional[str] = None,
    ) -> List[Document]:
        """Map CU ``AnalysisResult`` to LangChain ``Document`` objects."""
        documents: List[Document] = []

        for content_idx, content in enumerate(result.contents):
            if self._output_mode == "markdown":
                docs = self._map_markdown_mode(content, result)
            elif self._output_mode == "page":
                docs = self._map_page_mode(content, result)
            elif self._output_mode == "segment":
                docs = self._map_segment_mode(content, result)
            else:
                docs = []

            # Attach operation_id and set Document.id for tracing/dedup
            for doc in docs:
                if operation_id:
                    doc.metadata["operation_id"] = operation_id
                    if "page" in doc.metadata:
                        doc.id = (
                            f"{operation_id}_{content_idx}"
                            f"_page_{doc.metadata['page']}"
                        )
                    elif "segment_id" in doc.metadata:
                        doc.id = (
                            f"{operation_id}_{content_idx}"
                            f"_segment_{doc.metadata['segment_id']}"
                        )
                    else:
                        doc.id = f"{operation_id}_{content_idx}"

            documents.extend(docs)

        return documents

    def _build_base_metadata(
        self,
        content: Any,
        result: AnalysisResult,
    ) -> Dict[str, Any]:
        """Build metadata common to all output modes."""
        metadata: Dict[str, Any] = {
            "source": self._source,
            "mime_type": content.mime_type,
            "analyzer_id": getattr(content, "analyzer_id", None) or result.analyzer_id or self._analyzer_id,
            "output_mode": self._output_mode,
            "kind": content.kind,
        }

        if content.kind == "document":
            metadata["start_page_number"] = content.start_page_number
            metadata["end_page_number"] = content.end_page_number
        elif content.kind == "audioVisual":
            metadata["start_time_ms"] = content.start_time_ms
            metadata["end_time_ms"] = content.end_time_ms
            if content.width is not None:
                metadata["width"] = content.width
            if content.height is not None:
                metadata["height"] = content.height

        if content.fields and self._should_include_fields():
            metadata["fields"] = self._flatten_fields(content.fields)

        # Content-level classification result (if applicable)
        category = getattr(content, "category", None)
        if category:
            metadata["category"] = category

        return metadata

    def _should_include_fields(self) -> bool:
        """Check whether fields should be included in metadata."""
        if self._output_selection is None:
            return True
        return "fields" in self._output_selection

    # --- markdown mode ---

    def _map_markdown_mode(
        self,
        content: Any,
        result: AnalysisResult,
    ) -> List[Document]:
        """One ``Document`` per content item with full markdown text."""
        metadata = self._build_base_metadata(content, result)
        page_content = content.markdown or ""
        return [Document(page_content=page_content, metadata=metadata)]

    # --- page mode ---

    def _map_page_mode(
        self,
        content: Any,
        result: AnalysisResult,
    ) -> List[Document]:
        """One ``Document`` per page (documents only)."""
        if content.kind != "document":
            logger.warning(
                "output_mode='page' is not applicable to %s content. "
                "Falling back to 'markdown' mode.",
                content.kind,
            )
            return self._map_markdown_mode(content, result)

        if not content.pages:
            logger.warning(
                "No pages found in document content. "
                "Falling back to 'markdown' mode."
            )
            return self._map_markdown_mode(content, result)

        full_markdown = content.markdown or ""
        documents: List[Document] = []

        # Page-level Documents intentionally omit document-level fields and
        # category — these are document-wide, not page-specific.
        for page in content.pages:
            if not page.spans:
                logger.warning(
                    "Page %d has no spans. Falling back to 'markdown' mode.",
                    page.page_number,
                )
                return self._map_markdown_mode(content, result)

            page_text = self._extract_text_from_spans(full_markdown, page.spans)
            # Strip internal page-break markers
            page_text = page_text.replace("<!-- PageBreak -->", "").strip()

            metadata: Dict[str, Any] = {
                "source": self._source,
                "mime_type": content.mime_type,
                "analyzer_id": getattr(content, "analyzer_id", None) or result.analyzer_id or self._analyzer_id,
                "output_mode": self._output_mode,
                "kind": content.kind,
                "page": page.page_number,
            }
            documents.append(Document(page_content=page_text, metadata=metadata))

        return documents

    # --- segment mode ---

    def _map_segment_mode(
        self,
        content: Any,
        result: AnalysisResult,
    ) -> List[Document]:
        """One ``Document`` per content segment."""
        segments = getattr(content, "segments", None)

        if not segments:
            raise ValueError(
                "output_mode='segment' was requested but no segments were found "
                "in the analysis result. Ensure the analyzer has "
                "enableSegment=true with contentCategories defined."
            )

        full_markdown = content.markdown or ""
        documents: List[Document] = []

        for idx, segment in enumerate(segments):
            # Extract segment text
            segment_text = ""
            spans = getattr(segment, "spans", None)
            if spans:
                segment_text = self._extract_text_from_spans(full_markdown, spans)
            elif hasattr(segment, "markdown") and segment.markdown:
                segment_text = segment.markdown

            metadata: Dict[str, Any] = {
                "source": self._source,
                "mime_type": content.mime_type,
                "analyzer_id": getattr(content, "analyzer_id", None) or result.analyzer_id or self._analyzer_id,
                "output_mode": self._output_mode,
                "kind": content.kind,
                "segment_id": idx,
            }

            if hasattr(segment, "category") and segment.category:
                metadata["category"] = segment.category

            # Time range for audio/visual segments
            if hasattr(segment, "start_time_ms") and segment.start_time_ms is not None:
                metadata["start_time_ms"] = segment.start_time_ms
            if hasattr(segment, "end_time_ms") and segment.end_time_ms is not None:
                metadata["end_time_ms"] = segment.end_time_ms

            # Segment-level fields
            seg_fields = getattr(segment, "fields", None)
            if seg_fields and self._should_include_fields():
                metadata["fields"] = self._flatten_fields(seg_fields)

            documents.append(Document(page_content=segment_text, metadata=metadata))

        return documents

    # ------------------------------------------------------------------
    # Span / field helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_from_spans(full_text: str, spans: List[Any]) -> str:
        """Slice text from a list of ``ContentSpan`` objects."""
        parts: List[str] = []
        for span in spans:
            parts.append(full_text[span.offset : span.offset + span.length])
        return "".join(parts)

    def _flatten_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Convert CU ``ContentField`` objects to plain dicts with confidence."""
        result: Dict[str, Any] = {}
        for name, field in fields.items():
            result[name] = self._flatten_single_field(field)
        return result

    def _flatten_single_field(self, field: Any) -> Any:
        """Convert one ``ContentField`` to ``{type, value, confidence}``.

        Uses the SDK's ``.value`` convenience property which dynamically
        reads the correct ``value_*`` attribute for each field subclass.
        Object and array types are recursively flattened.
        """
        field_type = field.type

        # Array fields → list of {value, confidence}
        if field_type == "array" and field.value is not None:
            return [
                {
                    "value": self._resolve_field_value(item),
                    "confidence": item.confidence,
                }
                for item in field.value
            ]

        return {
            "type": field_type,
            "value": self._resolve_field_value(field),
            "confidence": field.confidence,
        }

    def _resolve_field_value(self, field: Any) -> Any:
        """Extract the plain Python value from a ``ContentField``.

        Uses the SDK's ``.value`` convenience property. For object fields,
        recursively resolves nested values. For array fields, returns a
        list of ``{value, confidence}`` dicts.
        """
        t = field.type
        raw = field.value

        if t == "object" and raw is not None:
            return {
                k: self._resolve_field_value(v) for k, v in raw.items()
            }
        if t == "array" and raw is not None:
            return [
                {
                    "value": self._resolve_field_value(item),
                    "confidence": item.confidence,
                }
                for item in raw
            ]
        # date/time .value already returns str; all others return native types
        return raw
