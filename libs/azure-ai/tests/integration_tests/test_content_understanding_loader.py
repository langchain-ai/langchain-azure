"""Integration tests for AzureAIContentUnderstandingLoader.

All sync tests use VCR cassettes for recording and replay.
Async tests require a live endpoint (VCR + aiohttp not supported here).

Usage:
    # Record cassettes (requires AZURE_CONTENT_UNDERSTANDING_ENDPOINT):
    pytest tests/integration_tests/test_content_understanding_loader.py \
        -v --record-mode=all -s

    # Replay from cassettes (no credentials needed):
    pytest tests/integration_tests/test_content_understanding_loader.py \
        -v --record-mode=none -s
"""

from __future__ import annotations

import os
from typing import Any, Dict, MutableMapping, cast
from urllib.parse import urlparse

import pytest
from vcr import VCR  # type: ignore[import-not-found, import-untyped]

from langchain_azure_ai.document_loaders.content_understanding import (
    AzureAIContentUnderstandingLoader,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENDPOINT = os.environ.get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "")
API_KEY = os.environ.get("AZURE_CONTENT_UNDERSTANDING_KEY", "")

# ---------------------------------------------------------------------------
# VCR redaction — protects credentials & endpoint hostnames in cassettes
# ---------------------------------------------------------------------------
_REDACTED_HOST = "redacted.services.ai.azure.com"
_REAL_HOST = urlparse(ENDPOINT).hostname or ""

_REQUEST_FILTER_HEADERS = [
    ("authorization", "REDACTED"),
    ("api-key", "REDACTED"),
    ("x-api-key", "REDACTED"),
    ("user-agent", "REDACTED"),
]

# Response headers that MUST be preserved for LRO replay to work.
_LRO_PRESERVE_HEADERS = {
    "operation-location",
    "content-type",
    "retry-after",
}


def _redact_host(text: str) -> str:
    """Replace the real endpoint hostname with a placeholder."""
    if _REAL_HOST:
        return text.replace(_REAL_HOST, _REDACTED_HOST)
    return text


def _sanitize_request(request: Any) -> Any:
    """Redact auth tokens, identifying info, and endpoint URIs from requests."""
    headers = cast(MutableMapping[str, Any], getattr(request, "headers", {}))
    for header, replacement in _REQUEST_FILTER_HEADERS:
        if header in headers:
            headers[header] = replacement
    if hasattr(request, "uri"):
        request.uri = _redact_host(request.uri)
    return request


def _sanitize_response_lro(response: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize response headers while preserving LRO-critical ones.

    Also redacts the endpoint hostname in Operation-Location URLs and
    response bodies.
    """
    headers = cast(MutableMapping[str, Any], response.get("headers", {}))
    for header in list(headers):
        if header.lower() in _LRO_PRESERVE_HEADERS:
            val = headers[header]
            if isinstance(val, list):
                headers[header] = [_redact_host(v) for v in val]
            elif isinstance(val, str):
                headers[header] = _redact_host(val)
            continue
        headers[header] = ["REDACTED"]
    body = response.get("body", {})
    if isinstance(body.get("string"), str):
        body["string"] = _redact_host(body["string"])
    return response


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    """Override shared vcr_config to preserve LRO headers and redact URIs."""
    return {
        "filter_headers": _REQUEST_FILTER_HEADERS,
        "match_on": ["method", "uri", "body"],
        "decode_compressed_response": True,
        "before_record_request": _sanitize_request,
        "before_record_response": _sanitize_response_lro,
        "path_transformer": VCR.ensure_suffix(".yaml"),
    }


# ---------------------------------------------------------------------------
# Asset URLs
# ---------------------------------------------------------------------------
_ASSET_BASE = (
    "https://raw.githubusercontent.com/Azure-Samples/"
    "azure-ai-content-understanding-assets/main"
)
SAMPLE_PDF_URL = f"{_ASSET_BASE}/document/invoice.pdf"
SAMPLE_IMAGE_URL = f"{_ASSET_BASE}/image/pieChart.jpg"
SAMPLE_AUDIO_URL = f"{_ASSET_BASE}/audio/callCenterRecording.mp3"
SAMPLE_VIDEO_URL = f"{_ASSET_BASE}/videos/sdk_samples/FlightSimulator.mp4"
MIXED_FINANCIAL_DOCS_URL = f"{_ASSET_BASE}/document/mixed_financial_docs.pdf"

# Deterministic analyzer IDs for VCR cassettes (must not vary between runs).
_VCR_CLASSIFIER_ID = "langchain_vcr_classifier"
_VCR_FIELDS_ID = "langchain_vcr_fields"
_VCR_SEGMENT_INV_ID = "langchain_vcr_seg_inv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_endpoint() -> str:
    """Return the real endpoint for recording, redacted endpoint for replay."""
    if _REAL_HOST:
        return ENDPOINT
    return f"https://{_REDACTED_HOST}/"


def _get_credential() -> Any:
    """Return API key, DefaultAzureCredential, or a placeholder for replay."""
    if API_KEY:
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential(API_KEY)
    try:
        from azure.identity import DefaultAzureCredential

        return DefaultAzureCredential()
    except ImportError:
        # In VCR replay mode, credential is never sent over the wire.
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential("PLACEHOLDER")


def _get_cu_client() -> Any:
    """Return a ContentUnderstandingClient for analyzer management."""
    from azure.ai.contentunderstanding import ContentUnderstandingClient

    return ContentUnderstandingClient(
        endpoint=_get_endpoint(), credential=_get_credential()
    )


# ---------------------------------------------------------------------------
# Recorded tests — sync document loading (prebuilt analyzers)
# ---------------------------------------------------------------------------
class TestDocumentLoading:
    """Sync integration tests — recorded with VCR."""

    @pytest.mark.vcr()
    def test_load_pdf_from_url_markdown_mode(self) -> None:
        """Load a PDF from URL in markdown mode."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=_get_endpoint(),
            credential=_get_credential(),
            analyzer_id="prebuilt-documentSearch",
            url=SAMPLE_PDF_URL,
            output_mode="markdown",
        )
        docs = loader.load()

        assert len(docs) >= 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["source"] == SAMPLE_PDF_URL
        assert docs[0].metadata["kind"] == "document"
        assert docs[0].metadata["output_mode"] == "markdown"
        assert docs[0].metadata["analyzer_id"] == "prebuilt-documentSearch"
        assert docs[0].metadata["mime_type"] == "application/pdf"
        assert isinstance(docs[0].metadata["start_page_number"], int)
        assert isinstance(docs[0].metadata["end_page_number"], int)

    @pytest.mark.vcr()
    def test_load_pdf_from_url_page_mode(self) -> None:
        """Load a PDF from URL in page mode."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=_get_endpoint(),
            credential=_get_credential(),
            analyzer_id="prebuilt-documentSearch",
            url=SAMPLE_PDF_URL,
            output_mode="page",
        )
        docs = loader.load()

        assert len(docs) >= 1
        for doc in docs:
            assert "page" in doc.metadata
            assert doc.metadata["output_mode"] == "page"
            assert doc.metadata["kind"] == "document"

    @pytest.mark.vcr()
    def test_load_image_from_url(self) -> None:
        """Load an image from URL."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=_get_endpoint(),
            credential=_get_credential(),
            url=SAMPLE_IMAGE_URL,
            output_mode="markdown",
        )
        docs = loader.load()

        assert len(docs) >= 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["source"] == SAMPLE_IMAGE_URL

    @pytest.mark.vcr()
    def test_load_audio_from_url(self) -> None:
        """Load audio from URL."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=_get_endpoint(),
            credential=_get_credential(),
            url=SAMPLE_AUDIO_URL,
            output_mode="markdown",
        )
        docs = loader.load()

        assert len(docs) >= 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["source"] == SAMPLE_AUDIO_URL
        assert docs[0].metadata["kind"] == "audioVisual"
        assert isinstance(docs[0].metadata["start_time_ms"], int)
        assert isinstance(docs[0].metadata["end_time_ms"], int)

    @pytest.mark.vcr()
    def test_load_video_from_url(self) -> None:
        """Load video from URL."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=_get_endpoint(),
            credential=_get_credential(),
            url=SAMPLE_VIDEO_URL,
            output_mode="markdown",
        )
        docs = loader.load()

        assert len(docs) >= 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["source"] == SAMPLE_VIDEO_URL
        assert docs[0].metadata["kind"] == "audioVisual"

    @pytest.mark.vcr()
    def test_load_invoice_with_field_extraction(self) -> None:
        """Load an invoice PDF with prebuilt-invoice and verify fields."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=_get_endpoint(),
            credential=_get_credential(),
            analyzer_id="prebuilt-invoice",
            url=SAMPLE_PDF_URL,
            output_mode="markdown",
        )
        docs = loader.load()

        assert len(docs) >= 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["kind"] == "document"
        assert "fields" in docs[0].metadata
        fields = docs[0].metadata["fields"]
        assert isinstance(fields, dict)
        assert len(fields) > 0
        for _name, field_val in fields.items():
            if isinstance(field_val, dict):
                assert "type" in field_val
                assert "value" in field_val
                assert "confidence" in field_val

    @pytest.mark.vcr()
    def test_operation_id_present(self) -> None:
        """Verify that operation_id is captured from the poller."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=_get_endpoint(),
            credential=_get_credential(),
            analyzer_id="prebuilt-documentSearch",
            url=SAMPLE_PDF_URL,
        )
        docs = loader.load()

        assert len(docs) >= 1
        assert "operation_id" in docs[0].metadata
        assert isinstance(docs[0].metadata["operation_id"], str)
        assert len(docs[0].metadata["operation_id"]) > 0
        assert docs[0].id is not None
        assert docs[0].metadata["operation_id"] in docs[0].id

    @pytest.mark.vcr()
    def test_page_mode_document_ids(self) -> None:
        """Verify Document.id format in page mode."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=_get_endpoint(),
            credential=_get_credential(),
            analyzer_id="prebuilt-documentSearch",
            url=SAMPLE_PDF_URL,
            output_mode="page",
        )
        docs = loader.load()

        assert len(docs) >= 1
        for doc in docs:
            assert doc.id is not None
            page_num = doc.metadata["page"]
            assert f"_page_{page_num}" in doc.id

    @pytest.mark.vcr()
    def test_page_mode_on_audio_falls_back_to_markdown(self) -> None:
        """Page mode on audio should fall back to markdown mode."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=_get_endpoint(),
            credential=_get_credential(),
            url=SAMPLE_AUDIO_URL,
            output_mode="page",
        )
        docs = loader.load()

        assert len(docs) == 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["kind"] == "audioVisual"

    @pytest.mark.vcr()
    def test_metadata_selection_excludes_fields(self) -> None:
        """When metadata_selection omits 'fields', metadata should not have fields."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=_get_endpoint(),
            credential=_get_credential(),
            analyzer_id="prebuilt-invoice",
            url=SAMPLE_PDF_URL,
            metadata_selection=["tables"],
        )
        docs = loader.load()

        assert len(docs) >= 1
        assert "fields" not in docs[0].metadata

    @pytest.mark.vcr()
    def test_custom_source_label(self) -> None:
        """Verify custom source label overrides URL in metadata."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=_get_endpoint(),
            credential=_get_credential(),
            url=SAMPLE_PDF_URL,
            source="my-custom-label.pdf",
        )
        docs = loader.load()

        assert docs[0].metadata["source"] == "my-custom-label.pdf"


# ---------------------------------------------------------------------------
# Recorded tests — custom analyzers (create / analyze / delete lifecycle)
# ---------------------------------------------------------------------------
class TestCustomAnalyzerIntegration:
    """Tests that create temporary custom analyzers and clean up after.

    Uses deterministic analyzer IDs so VCR cassettes have stable URIs.
    """

    @pytest.mark.vcr()
    def test_classifier_with_segment_mode(self) -> None:
        """Create a classifier with segmentation, analyze mixed_financial_docs.pdf,
        then verify segment mode returns multiple Documents with categories."""
        from azure.ai.contentunderstanding.models import (
            ContentAnalyzer,
            ContentAnalyzerConfig,
            ContentCategoryDefinition,
        )

        client = _get_cu_client()

        try:
            categories = {
                "Loan_Application": ContentCategoryDefinition(
                    description="Documents submitted by individuals or businesses "
                    "to request funding, including personal details, "
                    "financial history, and loan amount."
                ),
                "Invoice": ContentCategoryDefinition(
                    description="Billing documents issued by sellers or service "
                    "providers to request payment for goods or services."
                ),
                "Bank_Statement": ContentCategoryDefinition(
                    description="Official statements issued by banks that summarize "
                    "account activity over a period."
                ),
            }

            config = ContentAnalyzerConfig(
                return_details=True,
                enable_segment=True,
                content_categories=categories,
            )

            classifier = ContentAnalyzer(
                base_analyzer_id="prebuilt-document",
                description="LangChain test classifier",
                config=config,
                models={"completion": "gpt-4.1"},
            )

            poller = client.begin_create_analyzer(
                analyzer_id=_VCR_CLASSIFIER_ID,
                resource=classifier,
            )
            poller.result()

            loader = AzureAIContentUnderstandingLoader(
                endpoint=_get_endpoint(),
                credential=_get_credential(),
                analyzer_id=_VCR_CLASSIFIER_ID,
                url=MIXED_FINANCIAL_DOCS_URL,
                output_mode="segment",
            )
            docs = loader.load()

            assert len(docs) >= 2, f"Expected multiple segments, got {len(docs)}"
            for doc in docs:
                assert doc.metadata["output_mode"] == "segment"
                assert "segment_id" in doc.metadata
                assert isinstance(doc.metadata["segment_id"], int)
                assert "category" in doc.metadata

            categories_found = {doc.metadata["category"] for doc in docs}
            expected = {"Loan_Application", "Invoice", "Bank_Statement"}
            assert categories_found.issubset(
                expected
            ), f"Unexpected categories: {categories_found - expected}"

            for doc in docs:
                if doc.id:
                    assert f"_segment_{doc.metadata['segment_id']}" in doc.id

        finally:
            try:
                client.delete_analyzer(analyzer_id=_VCR_CLASSIFIER_ID)
            except Exception:
                pass

    @pytest.mark.vcr()
    def test_custom_field_extraction_analyzer(self) -> None:
        """Create a custom analyzer with field schema, analyze the invoice,
        and verify fields appear in metadata with correct structure."""
        from azure.ai.contentunderstanding.models import (
            ContentAnalyzer,
            ContentAnalyzerConfig,
            ContentFieldDefinition,
            ContentFieldSchema,
            ContentFieldType,
            GenerationMethod,
        )

        client = _get_cu_client()

        try:
            field_schema = ContentFieldSchema(
                name="invoice_schema",
                description="Schema for extracting invoice information",
                fields={
                    "vendor_name": ContentFieldDefinition(
                        type=ContentFieldType.STRING,
                        method=GenerationMethod.EXTRACT,
                        description="Name of the vendor or seller",
                        estimate_source_and_confidence=True,
                    ),
                    "total_amount": ContentFieldDefinition(
                        type=ContentFieldType.NUMBER,
                        method=GenerationMethod.EXTRACT,
                        description="Total amount on the invoice",
                        estimate_source_and_confidence=True,
                    ),
                    "document_type": ContentFieldDefinition(
                        type=ContentFieldType.STRING,
                        method=GenerationMethod.CLASSIFY,
                        description="Type of document",
                        enum=["invoice", "receipt", "contract", "other"],
                    ),
                },
            )

            config = ContentAnalyzerConfig(
                return_details=True,
                enable_ocr=True,
                estimate_field_source_and_confidence=True,
            )

            analyzer = ContentAnalyzer(
                base_analyzer_id="prebuilt-document",
                description="LangChain test field extraction analyzer",
                config=config,
                field_schema=field_schema,
                models={
                    "completion": "gpt-4.1",
                    "embedding": "text-embedding-3-large",
                },
            )

            poller = client.begin_create_analyzer(
                analyzer_id=_VCR_FIELDS_ID,
                resource=analyzer,
            )
            poller.result()

            loader = AzureAIContentUnderstandingLoader(
                endpoint=_get_endpoint(),
                credential=_get_credential(),
                analyzer_id=_VCR_FIELDS_ID,
                url=SAMPLE_PDF_URL,
                output_mode="markdown",
            )
            docs = loader.load()

            assert len(docs) >= 1
            assert len(docs[0].page_content) > 0
            assert "fields" in docs[0].metadata

            fields = docs[0].metadata["fields"]
            assert isinstance(fields, dict)

            assert "vendor_name" in fields
            vendor = fields["vendor_name"]
            assert isinstance(vendor, dict)
            assert "type" in vendor
            assert vendor["type"] == "string"
            assert "value" in vendor
            assert "confidence" in vendor
            assert isinstance(vendor["value"], str)
            assert len(vendor["value"]) > 0

            assert "total_amount" in fields
            total = fields["total_amount"]
            assert isinstance(total, dict)
            assert total["type"] == "number"
            assert isinstance(total["value"], (int, float))

            assert "document_type" in fields
            doc_type = fields["document_type"]
            assert isinstance(doc_type, dict)
            assert doc_type["value"] in [
                "invoice",
                "receipt",
                "contract",
                "other",
            ]

        finally:
            try:
                client.delete_analyzer(analyzer_id=_VCR_FIELDS_ID)
            except Exception:
                pass

    @pytest.mark.vcr()
    def test_segment_mode_on_single_page_invoice(self) -> None:
        """Create a classifier with segmentation, analyze a single-page
        invoice, and verify exactly one segment with category."""
        from azure.ai.contentunderstanding.models import (
            ContentAnalyzer,
            ContentAnalyzerConfig,
            ContentCategoryDefinition,
        )

        client = _get_cu_client()

        try:
            categories = {
                "Invoice": ContentCategoryDefinition(
                    description="Billing documents issued by sellers or service "
                    "providers to request payment for goods or services."
                ),
                "Report": ContentCategoryDefinition(
                    description="Analytical or summary documents presenting data, "
                    "findings, or recommendations."
                ),
            }

            config = ContentAnalyzerConfig(
                return_details=True,
                enable_segment=True,
                content_categories=categories,
            )

            classifier = ContentAnalyzer(
                base_analyzer_id="prebuilt-document",
                description="LangChain test segment single-page",
                config=config,
                models={"completion": "gpt-4.1"},
            )

            poller = client.begin_create_analyzer(
                analyzer_id=_VCR_SEGMENT_INV_ID,
                resource=classifier,
            )
            poller.result()

            loader = AzureAIContentUnderstandingLoader(
                endpoint=_get_endpoint(),
                credential=_get_credential(),
                analyzer_id=_VCR_SEGMENT_INV_ID,
                url=SAMPLE_PDF_URL,
                output_mode="segment",
            )
            docs = loader.load()

            assert len(docs) >= 1
            assert docs[0].metadata["output_mode"] == "segment"
            assert docs[0].metadata["category"] in ["Invoice", "Report"]
            assert "operation_id" in docs[0].metadata

        finally:
            try:
                client.delete_analyzer(analyzer_id=_VCR_SEGMENT_INV_ID)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Live-only async tests — VCR + aiohttp transport not supported here
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not ENDPOINT,
    reason="AZURE_CONTENT_UNDERSTANDING_ENDPOINT must be set",
)
class TestAsyncDocumentLoading:
    """Async tests require a live endpoint (VCR does not patch aiohttp)."""

    @pytest.mark.asyncio
    async def test_aload_pdf_from_url(self) -> None:
        """Async load a PDF from URL."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=ENDPOINT,
            credential=_get_credential(),
            analyzer_id="prebuilt-documentSearch",
            url=SAMPLE_PDF_URL,
        )
        docs = await loader.aload()

        assert len(docs) >= 1
        assert len(docs[0].page_content) > 0
        assert "operation_id" in docs[0].metadata

    @pytest.mark.asyncio
    async def test_aload_audio_from_url(self) -> None:
        """Async load audio from URL."""
        loader = AzureAIContentUnderstandingLoader(
            endpoint=ENDPOINT,
            credential=_get_credential(),
            url=SAMPLE_AUDIO_URL,
        )
        docs = await loader.aload()

        assert len(docs) >= 1
        assert docs[0].metadata["kind"] == "audioVisual"
        assert len(docs[0].page_content) > 0
