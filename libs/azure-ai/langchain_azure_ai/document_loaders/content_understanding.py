"""Azure Content Understanding document loader for LangChain."""

from __future__ import annotations

import logging
import mimetypes
import os

try:
    import filetype as _filetype
except ImportError:
    _filetype = None  # type: ignore[assignment]
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
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

from langchain_azure_ai.document_loaders._constants import (
    DEFAULT_ANALYZER,
    MEDIA_TYPE_ANALYZER_MAP,
    MIME_ALIASES,
)

logger = logging.getLogger(__name__)

_USER_AGENT = "langchain-azure-ai"


#: How to split CU results into LangChain ``Document`` objects.
#:
#: - ``"markdown"``: One document per content item with full markdown text.
#: - ``"page"``: One document per page (document content only).
#: - ``"segment"``: One document per content segment (requires a custom
#:   analyzer with ``enableSegment=true`` and ``contentCategories``).
OutputMode = Literal["markdown", "page", "segment"]

_VALID_OUTPUT_MODES = ("markdown", "page", "segment")


class AzureAIContentUnderstandingLoader(BaseLoader):
    """Load documents, images, audio, and video using Azure Content Understanding.

    Produces LangChain Document objects with extracted markdown content
    and rich metadata (fields, confidence scores, source info).

    Exactly one of ``file_path``, ``url``, or ``bytes_source`` must be provided.

    Example:
        .. code-block:: python

            from azure.identity import DefaultAzureCredential
            from langchain_azure_ai.document_loaders import (
                AzureAIContentUnderstandingLoader,
            )

            # Using a direct endpoint:
            loader = AzureAIContentUnderstandingLoader(
                endpoint="https://my-resource.services.ai.azure.com",
                credential=DefaultAzureCredential(),
                file_path="report.pdf",
            )

            # Using an Azure AI Foundry project endpoint:
            loader = AzureAIContentUnderstandingLoader(
                project_endpoint="https://my-resource.services.ai.azure.com/api/projects/my-project",
                credential=DefaultAzureCredential(),
                file_path="report.pdf",
            )
            docs = loader.load()
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Union[str, AzureKeyCredential, TokenCredential]] = None,
        *,
        project_endpoint: Optional[str] = None,
        analyzer_id: Optional[str] = None,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        bytes_source: Optional[bytes] = None,
        source: Optional[str] = None,
        output_mode: OutputMode = "markdown",
        content_range: Optional[str] = None,
        metadata_selection: Optional[List[str]] = None,
        model_deployments: Optional[Dict[str, str]] = None,
        analyze_kwargs: Optional[Dict[str, Any]] = None,
        api_version: Optional[str] = None,
    ) -> None:
        """Initialize the loader.

        Provide either ``endpoint`` or ``project_endpoint``, not both.
        When ``project_endpoint`` is given the base resource URL is
        derived automatically and ``credential`` must be a
        ``TokenCredential`` (e.g. ``DefaultAzureCredential``).

        If neither is provided, the loader checks the
        ``AZURE_AI_PROJECT_ENDPOINT`` environment variable.

        Args:
            endpoint: CU resource endpoint URL.
            credential: Azure credential — API key string,
                ``AzureKeyCredential``, or ``TokenCredential``.
            project_endpoint: Azure AI Foundry project endpoint URL
                (e.g. ``https://<resource>.services.ai.azure.com/api/projects/<project>``).
                Mutually exclusive with ``endpoint``. Falls back to the
                ``AZURE_AI_PROJECT_ENDPOINT`` environment variable.
            analyzer_id: Analyzer to use. Defaults by input MIME type if omitted.
            file_path: Path to a local file (mutually exclusive with
                ``url`` and ``bytes_source``).
            url: Publicly accessible URL pointing to the content
                (mutually exclusive with ``file_path`` and
                ``bytes_source``).
            bytes_source: Raw bytes of the content, e.g. from an
                in-memory download, database blob, or Azure Blob
                Storage response (mutually exclusive with ``file_path``
                and ``url``).
            source: Label for ``metadata["source"]``. Defaults to *file_path*
                or *url* when provided.
            output_mode: How to split results into Documents —
                ``"markdown"`` (default), ``"page"``,
                or ``"segment"``. Segment mode requires a custom
                analyzer with ``enableSegment=true`` and
                ``contentCategories``. Supported for document and video
                analyzers only — audio-based analyzers do not support
                segmentation.
            content_range: Subset of input to analyze. Pages use 1-based
                page numbers, e.g. ``"1-3,5,9-"`` (page 1 through 3,
                page 5, and page 9 onward); audio/video uses
                milliseconds ``"0-60000"``.
            metadata_selection: What to include in metadata, e.g.
                ``["fields", "tables"]``. Defaults to include fields.
            model_deployments: Optional mapping of model names to
                deployment names. Use this to override default model
                deployments for custom analyzers.
            analyze_kwargs: Extra keyword arguments forwarded to
                ``begin_analyze`` (e.g., ``processing_location``).
            api_version: Content Understanding API version to use
                (e.g. ``"2025-11-01"``). Defaults to the latest version
                supported by the installed SDK.
        """
        endpoint, credential = self._resolve_endpoint(
            endpoint=endpoint,
            credential=credential,
            project_endpoint=project_endpoint,
        )

        if not endpoint or not str(endpoint).strip():
            raise ValueError(
                "An endpoint is required. Provide 'endpoint' or "
                "'project_endpoint', or set the "
                "AZURE_AI_PROJECT_ENDPOINT environment variable."
            )

        sources = [file_path, url, bytes_source]
        if sum(s is not None for s in sources) != 1:
            raise ValueError(
                "Exactly one of file_path, url, or bytes_source must be provided."
            )

        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of "
                f"{list(_VALID_OUTPUT_MODES)}, "
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
        self._metadata_selection = metadata_selection
        self._model_deployments = model_deployments
        self._analyze_kwargs = analyze_kwargs or {}
        self._api_version = api_version

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

    @staticmethod
    def _resolve_endpoint(
        *,
        endpoint: Optional[str],
        credential: Optional[Union[str, AzureKeyCredential, TokenCredential]],
        project_endpoint: Optional[str],
    ) -> tuple:
        """Resolve endpoint and credential from direct or project endpoint.

        When ``project_endpoint`` is provided (or discovered via
        ``AZURE_AI_PROJECT_ENDPOINT``), the base resource URL is extracted
        and only ``TokenCredential`` is accepted.
        """
        if endpoint and project_endpoint:
            raise ValueError(
                "'endpoint' and 'project_endpoint' are mutually exclusive. "
                "Provide only one."
            )

        # Fall back to env var when neither is provided
        if not endpoint and not project_endpoint:
            project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")

        if project_endpoint:
            if not isinstance(credential, TokenCredential):
                if credential is None:
                    try:
                        from azure.identity import DefaultAzureCredential

                        credential = DefaultAzureCredential()
                        logger.info(
                            "No credential provided with project_endpoint; "
                            "using DefaultAzureCredential."
                        )
                    except ImportError:
                        raise ImportError(
                            "'azure-identity' is required when using "
                            "'project_endpoint' without an explicit credential. "
                            "Install with: pip install azure-identity"
                        )
                else:
                    raise ValueError(
                        "When using 'project_endpoint', the credential must "
                        "be a TokenCredential (e.g. DefaultAzureCredential), "
                        "not an API key."
                    )

            # Strip /api/projects/<project> to get the base resource URL
            base = project_endpoint.split("/api/projects")[0].rstrip("/")
            return base, credential

        return endpoint, credential

    def lazy_load(self) -> Iterator[Document]:
        """Load documents synchronously.

        Yields:
            ``Document`` objects parsed from the CU analysis result.
        """
        client_kwargs: Dict[str, Any] = {"user_agent": _USER_AGENT}
        if self._api_version is not None:
            client_kwargs["api_version"] = self._api_version
        client = ContentUnderstandingClient(
            endpoint=self._endpoint,
            credential=self._credential,  # type: ignore[arg-type]
            **client_kwargs,
        )

        try:
            poller = self._start_analyze(client)
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

        client_kwargs: Dict[str, Any] = {"user_agent": _USER_AGENT}
        if self._api_version is not None:
            client_kwargs["api_version"] = self._api_version
        client = AsyncContentUnderstandingClient(
            endpoint=self._endpoint,
            credential=self._credential,  # type: ignore[arg-type]
            **client_kwargs,
        )
        try:
            poller = await self._start_analyze_async(client)
            operation_id: Optional[str] = getattr(poller, "operation_id", None)
            if not isinstance(operation_id, str):
                operation_id = None
            result: AnalysisResult = await poller.result()

            if isinstance(result.warnings, list):
                for warning in result.warnings:
                    logger.warning("CU analysis warning: %s", warning.message)

            if not result.contents:
                logger.warning("CU analysis returned no content items.")

            for doc in self._map_result_to_documents(result, operation_id=operation_id):
                yield doc
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    def _detect_mime_type(self) -> Optional[str]:
        """Detect MIME type using a three-layer strategy.

        1. **Extension-based** — ``mimetypes.guess_type`` on file path or URL.
        2. **Binary sniffing** — when step 1 returns ``None`` or
           ``application/octet-stream``, inspect magic bytes via the
           ``filetype`` library (optional dependency).  Only the first
           261 bytes are read.
        3. **Alias normalization** — map variant MIMEs (e.g.
           ``audio/x-wav``) to CU's canonical set via
           :data:`~._constants.MIME_ALIASES`.
        """
        # Layer 1: extension-based detection
        path = self._file_path or self._url
        if path:
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type and mime_type != "application/octet-stream":
                return MIME_ALIASES.get(mime_type, mime_type)

        # Layer 2: binary sniffing (when extension is missing or unhelpful)
        if _filetype is not None:
            sample: Optional[bytes] = None
            if self._bytes_source is not None:
                sample = self._bytes_source[:261]
            elif self._file_path:
                try:
                    with open(self._file_path, "rb") as f:
                        sample = f.read(261)
                except OSError:
                    pass
            if sample:
                kind = _filetype.guess(sample)
                if kind is not None:
                    return MIME_ALIASES.get(kind.mime, kind.mime)

        return None

    def _resolve_default_analyzer(self) -> str:
        """Pick a default analyzer based on MIME type prefix."""
        if self._mime_type:
            for prefix, analyzer in MEDIA_TYPE_ANALYZER_MAP.items():
                if self._mime_type.startswith(prefix):
                    return analyzer
        return DEFAULT_ANALYZER

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

    def _get_binary_data(self) -> Optional[bytes]:
        """Return raw bytes for binary upload, or ``None`` for URL inputs."""
        if self._file_path:
            with open(self._file_path, "rb") as f:
                return f.read()
        if self._bytes_source is not None:
            return self._bytes_source
        return None

    def _start_analyze(self, client: ContentUnderstandingClient) -> Any:
        """Start an analyze operation, choosing the optimal upload path.

        Uses ``begin_analyze_binary`` for local file / bytes inputs to
        avoid the ~33 % overhead of base64 encoding.  Falls back to
        ``begin_analyze`` (JSON body) when a URL is provided or when
        ``model_deployments`` is set (only supported on the JSON path).
        """
        binary_data = self._get_binary_data()
        if binary_data is not None and not self._model_deployments:
            return client.begin_analyze_binary(
                analyzer_id=self._analyzer_id,
                binary_input=binary_data,
                content_range=self._content_range,
                **self._analyze_kwargs,
            )
        analysis_input = self._build_analysis_input()
        return client.begin_analyze(
            analyzer_id=self._analyzer_id,
            inputs=[analysis_input],
            model_deployments=self._model_deployments,
            **self._analyze_kwargs,
        )

    async def _start_analyze_async(self, client: Any) -> Any:
        """Async version of :meth:`_start_analyze`."""
        binary_data = self._get_binary_data()
        if binary_data is not None and not self._model_deployments:
            return await client.begin_analyze_binary(
                analyzer_id=self._analyzer_id,
                binary_input=binary_data,
                content_range=self._content_range,
                **self._analyze_kwargs,
            )
        analysis_input = self._build_analysis_input()
        return await client.begin_analyze(
            analyzer_id=self._analyzer_id,
            inputs=[analysis_input],
            model_deployments=self._model_deployments,
            **self._analyze_kwargs,
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
                docs = self._map_segment_mode(content, result, content_idx)
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

        if self._output_mode == "segment" and not documents:
            raise ValueError(
                "output_mode='segment' was requested but no segments were found "
                "in the analysis result. Ensure the analyzer has "
                "enableSegment=true with contentCategories defined. "
                "Note: segmentation is only supported for document and video "
                "analyzers — audio-based analyzers (prebuilt-audio, "
                "prebuilt-callCenter) do not support enableSegment."
            )

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
            "analyzer_id": getattr(content, "analyzer_id", None)
            or result.analyzer_id
            or self._analyzer_id,
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
        if self._metadata_selection is None:
            return True
        return "fields" in self._metadata_selection

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
                "analyzer_id": getattr(content, "analyzer_id", None)
                or result.analyzer_id
                or self._analyzer_id,
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
        content_idx: int = 0,
    ) -> List[Document]:
        """One ``Document`` per content segment.

        Classification analyzers return sub-contents with ``category`` set
        (e.g. ``path="input1/segment1"``).  These are self-contained results
        from the sub-analyzer with their own markdown, page numbers, and
        fields — so we use them directly.

        For video segments where the parent content carries a ``segments``
        array with time ranges and spans, we extract text from the parent
        markdown via span offsets.

        Note: Segmentation is supported for **document** and **video**
        analyzers only.  Audio-based analyzers (``prebuilt-audio``,
        ``prebuilt-callCenter``) do not support ``enableSegment``.
        """
        # --- classified sub-contents (document / video classification) ---
        category = getattr(content, "category", None)
        if category:
            cat_meta = self._build_base_metadata(content, result)
            return [
                Document(
                    page_content=content.markdown or "",
                    metadata=cat_meta,
                )
            ]

        # --- parent content with a segments array (audio/visual) ---
        segments = getattr(content, "segments", None)
        if segments:
            # If sub-contents with ``category`` exist they are the authoritative
            # segment documents (with their own markdown, page numbers, fields).
            # Skip the parent's span-based extraction to avoid duplicates.
            has_sub_contents = any(
                getattr(c, "category", None)
                for c in result.contents
                if c is not content
            )
            if has_sub_contents:
                return []

            full_markdown = content.markdown or ""
            documents: List[Document] = []

            for idx, segment in enumerate(segments):
                # Extract segment text
                segment_text = ""
                # Both DocumentContentSegment and AudioVisualContentSegment
                # expose a single "span" attribute (not "spans").
                single_span = getattr(segment, "span", None)
                if single_span:
                    segment_text = self._extract_text_from_spans(
                        full_markdown, [single_span]
                    )
                elif hasattr(segment, "markdown") and segment.markdown:
                    segment_text = segment.markdown

                metadata: Dict[str, Any] = {
                    "source": self._source,
                    "mime_type": content.mime_type,
                    "analyzer_id": getattr(content, "analyzer_id", None)
                    or result.analyzer_id
                    or self._analyzer_id,
                    "output_mode": self._output_mode,
                    "kind": content.kind,
                    "segment_id": idx,
                }

                if hasattr(segment, "category") and segment.category:
                    metadata["category"] = segment.category

                # Time range for audio/visual segments
                if (
                    hasattr(segment, "start_time_ms")
                    and segment.start_time_ms is not None
                ):
                    metadata["start_time_ms"] = segment.start_time_ms
                if hasattr(segment, "end_time_ms") and segment.end_time_ms is not None:
                    metadata["end_time_ms"] = segment.end_time_ms

                # Segment-level fields
                seg_fields = getattr(segment, "fields", None)
                if seg_fields and self._should_include_fields():
                    metadata["fields"] = self._flatten_fields(seg_fields)

                documents.append(Document(page_content=segment_text, metadata=metadata))

            return documents

        # --- standalone segment content ---
        # Video segmenters may return sub-analyzer results as standalone
        # content items without ``category`` or a ``segments`` array.
        # Treat such content as an individual segment document.
        metadata = self._build_base_metadata(content, result)
        metadata["segment_id"] = content_idx
        return [Document(page_content=content.markdown or "", metadata=metadata)]

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
                    "confidence": getattr(item, "confidence", None),
                }
                for item in field.value
            ]

        return {
            "type": field_type,
            "value": self._resolve_field_value(field),
            "confidence": getattr(field, "confidence", None),
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
            return {k: self._resolve_field_value(v) for k, v in raw.items()}
        if t == "array" and raw is not None:
            return [
                {
                    "value": self._resolve_field_value(item),
                    "confidence": getattr(item, "confidence", None),
                }
                for item in raw
            ]
        # date/time .value already returns str; all others return native types
        return raw
