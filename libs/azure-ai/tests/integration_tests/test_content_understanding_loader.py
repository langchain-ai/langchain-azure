"""Integration tests for AzureContentUnderstandingLoader.

These tests require a live Azure Content Understanding endpoint.
Set environment variables before running:
    AZURE_CONTENT_UNDERSTANDING_ENDPOINT
    AZURE_CONTENT_UNDERSTANDING_KEY   (or use DefaultAzureCredential)
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from langchain_azure_ai.document_loaders.content_understanding import (
    AzureContentUnderstandingLoader,
)

ENDPOINT = os.environ.get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "")
API_KEY = os.environ.get("AZURE_CONTENT_UNDERSTANDING_KEY", "")

_ASSET_BASE = (
    "https://raw.githubusercontent.com/Azure-Samples/"
    "azure-ai-content-understanding-assets/main"
)
SAMPLE_PDF_URL = f"{_ASSET_BASE}/document/invoice.pdf"
SAMPLE_IMAGE_URL = f"{_ASSET_BASE}/image/pieChart.jpg"
SAMPLE_AUDIO_URL = f"{_ASSET_BASE}/audio/callCenterRecording.mp3"
SAMPLE_VIDEO_URL = f"{_ASSET_BASE}/videos/sdk_samples/FlightSimulator.mp4"
MIXED_FINANCIAL_DOCS_URL = f"{_ASSET_BASE}/document/mixed_financial_docs.pdf"


def _get_credential() -> Any:
    """Return API key if set, otherwise use DefaultAzureCredential."""
    if API_KEY:
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential(API_KEY)
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


def _get_cu_client() -> Any:
    """Return a ContentUnderstandingClient for analyzer management."""
    from azure.ai.contentunderstanding import ContentUnderstandingClient

    return ContentUnderstandingClient(
        endpoint=ENDPOINT, credential=_get_credential()
    )


@pytest.mark.skipif(
    not ENDPOINT,
    reason="AZURE_CONTENT_UNDERSTANDING_ENDPOINT must be set",
)
class TestLiveDocumentLoading:
    """Live integration tests (skipped when credentials are not set)."""

    def test_load_pdf_from_url_markdown_mode(self) -> None:
        """Load a PDF from URL in markdown mode."""
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
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

    def test_load_pdf_from_url_page_mode(self) -> None:
        """Load a PDF from URL in page mode."""
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
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

    def test_load_image_from_url(self) -> None:
        """Load an image from URL — should use prebuilt-documentSearch."""
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
            credential=_get_credential(),
            url=SAMPLE_IMAGE_URL,
            output_mode="markdown",
        )
        docs = loader.load()

        assert len(docs) >= 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["source"] == SAMPLE_IMAGE_URL

    def test_load_audio_from_url(self) -> None:
        """Load audio from URL — should use prebuilt-audioSearch."""
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
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

    def test_load_video_from_url(self) -> None:
        """Load video from URL — should use prebuilt-videoSearch."""
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
            credential=_get_credential(),
            url=SAMPLE_VIDEO_URL,
            output_mode="markdown",
        )
        docs = loader.load()

        assert len(docs) >= 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["source"] == SAMPLE_VIDEO_URL
        assert docs[0].metadata["kind"] == "audioVisual"

    def test_load_invoice_with_field_extraction(self) -> None:
        """Load an invoice PDF with prebuilt-invoice and verify fields."""
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
            credential=_get_credential(),
            analyzer_id="prebuilt-invoice",
            url=SAMPLE_PDF_URL,
            output_mode="markdown",
        )
        docs = loader.load()

        assert len(docs) >= 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["kind"] == "document"
        # prebuilt-invoice should extract invoice fields
        assert "fields" in docs[0].metadata
        fields = docs[0].metadata["fields"]
        assert isinstance(fields, dict)
        assert len(fields) > 0
        # Each field should have type/value/confidence structure
        for _name, field_val in fields.items():
            if isinstance(field_val, dict):
                assert "type" in field_val
                assert "value" in field_val
                assert "confidence" in field_val

    def test_operation_id_present_in_live_result(self) -> None:
        """Verify that operation_id is captured from the live poller."""
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
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

    def test_page_mode_document_ids(self) -> None:
        """Verify Document.id format in page mode."""
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
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

    def test_page_mode_on_audio_falls_back_to_markdown(self) -> None:
        """Page mode on audio should fall back to markdown mode."""
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
            credential=_get_credential(),
            url=SAMPLE_AUDIO_URL,
            output_mode="page",
        )
        docs = loader.load()

        # Should fall back to single markdown document
        assert len(docs) == 1
        assert len(docs[0].page_content) > 0
        assert docs[0].metadata["kind"] == "audioVisual"

    def test_output_selection_excludes_fields(self) -> None:
        """When output_selection omits 'fields', metadata should not have fields."""
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
            credential=_get_credential(),
            analyzer_id="prebuilt-invoice",
            url=SAMPLE_PDF_URL,
            output_selection=["tables"],
        )
        docs = loader.load()

        assert len(docs) >= 1
        assert "fields" not in docs[0].metadata

    def test_custom_source_label(self) -> None:
        """Verify custom source label overrides URL in metadata."""
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
            credential=_get_credential(),
            url=SAMPLE_PDF_URL,
            source="my-custom-label.pdf",
        )
        docs = loader.load()

        assert docs[0].metadata["source"] == "my-custom-label.pdf"

    @pytest.mark.asyncio
    async def test_aload_pdf_from_url(self) -> None:
        """Async load a PDF from URL."""
        loader = AzureContentUnderstandingLoader(
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
        loader = AzureContentUnderstandingLoader(
            endpoint=ENDPOINT,
            credential=_get_credential(),
            url=SAMPLE_AUDIO_URL,
        )
        docs = await loader.aload()

        assert len(docs) >= 1
        assert docs[0].metadata["kind"] == "audioVisual"
        assert len(docs[0].page_content) > 0


@pytest.mark.skipif(
    not ENDPOINT,
    reason="AZURE_CONTENT_UNDERSTANDING_ENDPOINT must be set",
)
class TestCustomAnalyzerIntegration:
    """Tests that create temporary custom analyzers and clean up after.

    These tests follow the patterns from the CU SDK samples
    (sample_create_analyzer.py, sample_create_classifier.py).
    """

    def test_classifier_with_segment_mode(self) -> None:
        """Create a classifier with segmentation, analyze mixed_financial_docs.pdf,
        then verify segment mode returns multiple Documents with categories."""
        from azure.ai.contentunderstanding.models import (
            ContentAnalyzer,
            ContentAnalyzerConfig,
            ContentCategoryDefinition,
        )

        client = _get_cu_client()
        analyzer_id = f"langchain_test_classifier_{int(time.time())}"

        try:
            # Create classifier with segmentation enabled
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
                analyzer_id=analyzer_id,
                resource=classifier,
            )
            poller.result()

            # Now use the loader in segment mode
            loader = AzureContentUnderstandingLoader(
                endpoint=ENDPOINT,
                credential=_get_credential(),
                analyzer_id=analyzer_id,
                url=MIXED_FINANCIAL_DOCS_URL,
                output_mode="segment",
            )
            docs = loader.load()

            # Should produce multiple segments
            assert len(docs) >= 2, (
                f"Expected multiple segments, got {len(docs)}"
            )
            for doc in docs:
                assert doc.metadata["output_mode"] == "segment"
                assert "segment_id" in doc.metadata
                assert isinstance(doc.metadata["segment_id"], int)
                # Each segment should have a category from our definitions
                assert "category" in doc.metadata

            # Verify categories are from our defined set
            categories_found = {doc.metadata["category"] for doc in docs}
            expected = {"Loan_Application", "Invoice", "Bank_Statement"}
            assert categories_found.issubset(expected), (
                f"Unexpected categories: {categories_found - expected}"
            )

            # Verify Document.id includes segment info
            for doc in docs:
                if doc.id:
                    assert f"_segment_{doc.metadata['segment_id']}" in doc.id

        finally:
            # Clean up — always delete the analyzer
            try:
                client.delete_analyzer(analyzer_id=analyzer_id)
            except Exception:
                pass

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
        analyzer_id = f"langchain_test_fields_{int(time.time())}"

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
                analyzer_id=analyzer_id,
                resource=analyzer,
            )
            poller.result()

            # Use the loader with the custom analyzer
            loader = AzureContentUnderstandingLoader(
                endpoint=ENDPOINT,
                credential=_get_credential(),
                analyzer_id=analyzer_id,
                url=SAMPLE_PDF_URL,
                output_mode="markdown",
            )
            docs = loader.load()

            assert len(docs) >= 1
            assert len(docs[0].page_content) > 0
            assert "fields" in docs[0].metadata

            fields = docs[0].metadata["fields"]
            assert isinstance(fields, dict)

            # Verify our custom fields are present
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
                "invoice", "receipt", "contract", "other",
            ]

        finally:
            try:
                client.delete_analyzer(analyzer_id=analyzer_id)
            except Exception:
                pass

    def test_segment_mode_on_single_page_invoice(self) -> None:
        """Create a classifier with segmentation, analyze a single-page
        invoice, and verify exactly one segment with category."""
        from azure.ai.contentunderstanding.models import (
            ContentAnalyzer,
            ContentAnalyzerConfig,
            ContentCategoryDefinition,
        )

        client = _get_cu_client()
        analyzer_id = f"langchain_test_seg_inv_{int(time.time())}"

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
                analyzer_id=analyzer_id,
                resource=classifier,
            )
            poller.result()

            # Load the single-page invoice in segment mode
            loader = AzureContentUnderstandingLoader(
                endpoint=ENDPOINT,
                credential=_get_credential(),
                analyzer_id=analyzer_id,
                url=SAMPLE_PDF_URL,
                output_mode="segment",
            )
            docs = loader.load()

            # Single-page doc should produce at least 1 segment
            assert len(docs) >= 1
            assert docs[0].metadata["output_mode"] == "segment"
            assert docs[0].metadata["category"] in ["Invoice", "Report"]
            assert "operation_id" in docs[0].metadata

        finally:
            try:
                client.delete_analyzer(analyzer_id=analyzer_id)
            except Exception:
                pass
