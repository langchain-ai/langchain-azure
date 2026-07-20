# Azure Content Understanding — LangChain Document Loader

Load and extract content from **documents, images, audio, and video** using
[Azure Content Understanding](https://learn.microsoft.com/azure/ai-services/content-understanding/).
The loader returns LangChain `Document` objects with clean markdown content and
rich metadata (fields, confidence scores, source info) — ready to use in RAG
pipelines and agent chains.

## Why Content Understanding?

Content Understanding turns messy, multimodal content — PDFs, Office
documents, images, and audio — into clean, structured, agent-ready output.
The loader loads the content, then Content Understanding **parses, extracts,
and classifies (segments)** it into markdown and JSON with grounded key-value
fields your agent can act on instead of raw bytes.

- **State-of-the-art multi-lingual layout parsing** — Content Understanding
  combines the proven traditional AI of Azure Document Intelligence with
  LLM-based reasoning, built on Microsoft's OCR and layout technology
  refined over 20+ years. It preserves structural elements — tables, headings,
  paragraphs, columns, selection marks and checkboxes, barcodes, embedded
  images — and handles a broad set of languages, often outperforming
  open-source alternatives on real-world documents.
- **Multi-modal in one loader** — Documents, images, audio, and video all go
  through the same `AzureAIContentUnderstandingLoader` interface. The
  appropriate prebuilt analyzer is auto-selected per file type when you don't
  specify one.
- **Structured field extraction** — [Prebuilt](https://learn.microsoft.com/azure/ai-services/content-understanding/concepts/prebuilt-analyzers)
  analyzers extract common domain-specific fields (e.g., invoice amounts,
  receipt dates, contract clauses) out of the box, with confidence scores
  and source grounding surfaced on `Document.metadata`. Build
  [custom analyzers](https://learn.microsoft.com/azure/ai-services/content-understanding/tutorial/create-custom-analyzer)
  when you need your own schema.
- **Chart and figure understanding** — The `prebuilt-documentSearch` analyzer
  extracts semantic content from charts and figures (e.g., bar chart values,
  trend descriptions), not just scattered axis labels.
- **Accurate table extraction** — CU produces correct markdown tables even when
  cells are empty. Most PDF text extractors lose column alignment and output a
  flat list of values, making it impossible to reconstruct which value belongs
  to which column. See this
  [sample PDF](https://github.com/Azure-Samples/azure-ai-content-understanding-assets/blob/main/document/financial_table_and_chart.pdf)
  for a table that breaks standard loaders.

## Installation

```bash
pip install -U langchain-azure-ai
```

The Content Understanding loader is included in the base package — no extras
required.

## Prerequisites

- An Azure AI Foundry project **or** an Azure Content Understanding resource.
  Follow the
  [Content Understanding Studio quickstart prerequisites](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/quickstart/content-understanding-studio?tabs=portal%2Ccu-studio#prerequisites)
  to create one.
- A credential: either a `TokenCredential` such as `DefaultAzureCredential()`
  (recommended) or an API key. Foundry project endpoints require a
  `TokenCredential`.

## Quick start

Load a PDF from a local file:

```python
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.document_loaders import AzureAIContentUnderstandingLoader

loader = AzureAIContentUnderstandingLoader(
    endpoint="https://{your-resource-name}.services.ai.azure.com",
    credential=DefaultAzureCredential(),
    file_path="report.pdf",
)

docs = loader.load()
print(docs[0].page_content[:200])   # markdown content
print(docs[0].metadata["source"])   # source file
```

You can also point the loader at a URL or raw bytes:

```python
# From a public URL
loader = AzureAIContentUnderstandingLoader(
    endpoint="https://{your-resource-name}.services.ai.azure.com",
    credential=DefaultAzureCredential(),
    url="https://raw.githubusercontent.com/Azure-Samples/azure-ai-content-understanding-assets/main/document/financial_table_and_chart.pdf",
)

# From raw bytes (e.g., an Azure Blob Storage download)
loader = AzureAIContentUnderstandingLoader(
    endpoint="https://{your-resource-name}.services.ai.azure.com",
    credential=DefaultAzureCredential(),
    bytes_source=blob_bytes,
)
```

If you're using an Azure AI Foundry project, pass `project_endpoint` instead of
`endpoint` — the loader resolves the underlying resource URL automatically:

```python
loader = AzureAIContentUnderstandingLoader(
    project_endpoint="https://{your-resource-name}.services.ai.azure.com/api/projects/{your-project}",
    credential=DefaultAzureCredential(),
    file_path="report.pdf",
)
```

## Structured field extraction

Use a prebuilt analyzer to extract domain-specific fields as metadata. Some
analyzers (like `prebuilt-invoice`) call a chat model under the hood — pass
`model_deployments` to map the model name to a deployment name in your
resource:

```python
loader = AzureAIContentUnderstandingLoader(
    endpoint="https://{your-resource-name}.services.ai.azure.com",
    credential=DefaultAzureCredential(),
    file_path="invoice.pdf",
    analyzer_id="prebuilt-invoice",
    model_deployments={"gpt-5.2": "gpt-5.2"},  # model name -> your deployment
)

docs = loader.load()
fields = docs[0].metadata.get("fields", {})

# Iterate all extracted fields
for name, data in fields.items():
    if isinstance(data, dict):
        print(f"{name}: {data.get('value')}  "
              f"(confidence: {data.get('confidence', 'N/A')})")
```

Common prebuilt analyzers:

| Analyzer ID | Use case |
|-------------|----------|
| `prebuilt-read` | OCR only — fastest option; no LLM, no model deployment required |
| `prebuilt-layout` | OCR plus layout structure (tables, headings, sections) — no LLM required |
| `prebuilt-documentSearch` (default) | Rich document / image extraction with charts and figures (LLM-powered) |
| `prebuilt-invoice` | Invoices — extracts vendor, totals, line items, dates |
| `prebuilt-audioSearch` | Audio — transcription with speaker + timing metadata |
| `prebuilt-videoSearch` | Video — segmented transcription with keyframes |

See the [prebuilt analyzers reference](https://learn.microsoft.com/azure/ai-services/content-understanding/concepts/prebuilt-analyzers)
for the full list, or build your own [custom analyzer](https://learn.microsoft.com/azure/ai-services/content-understanding/tutorial/create-custom-analyzer).

## Multi-modal: audio and video

The same loader handles audio and video — the appropriate prebuilt analyzer is
auto-selected when you don't pass `analyzer_id`:

```python
# Transcribe an audio file
loader = AzureAIContentUnderstandingLoader(
    endpoint="https://{your-resource-name}.services.ai.azure.com",
    credential=DefaultAzureCredential(),
    url="https://raw.githubusercontent.com/Azure-Samples/azure-ai-content-understanding-assets/main/audio/callCenterRecording.mp3",
)

docs = loader.load()
print(docs[0].page_content[:200])           # transcript as markdown
print(docs[0].metadata["start_time_ms"])    # 0
print(docs[0].metadata["end_time_ms"])      # total duration in ms
```

## Output modes

Control how results are split into `Document` objects with `output_mode`:

- `"markdown"` (default) — one document per content item with full markdown text.
- `"page"` — one document per page (document analyzers only).
- `"segment"` — one document per content segment. Requires a custom analyzer
  configured with `enableSegment=true` and `contentCategories`. Supported for
  document and video analyzers.

```python
loader = AzureAIContentUnderstandingLoader(
    endpoint="https://{your-resource-name}.services.ai.azure.com",
    credential=DefaultAzureCredential(),
    file_path="report.pdf",
    output_mode="page",   # one Document per page
)
```

## Key options

| Parameter | Description |
|-----------|-------------|
| `endpoint` / `project_endpoint` | Provide one. `project_endpoint` derives the resource URL and requires a `TokenCredential`. Falls back to `AZURE_AI_PROJECT_ENDPOINT`. |
| `credential` | API key string, `AzureKeyCredential`, or `TokenCredential` (e.g. `DefaultAzureCredential`). |
| `file_path` / `url` / `bytes_source` | Provide **exactly one** — the input source. |
| `analyzer_id` | Analyzer ID (prebuilt or custom). Auto-selected from MIME type when omitted. |
| `output_mode` | `"markdown"` (default), `"page"`, or `"segment"`. |
| `content_range` | Subset of input to analyze. Pages: 1-based (`"1-3,5,9-"`). Audio/video: milliseconds (`"0-60000"`). |
| `metadata_selection` | Which metadata categories to include, e.g. `["fields", "tables"]`. Defaults to `["fields"]`. |
| `model_deployments` | Mapping from model name → deployment name for custom analyzers that reference specific deployments. |

For the full signature and all options, see the class docstring in
[`content_understanding.py`](./content_understanding.py).

## Also available as an agent tool

If you want to give an LLM agent the ability to analyze documents on demand,
Content Understanding is also exposed as an agent tool via
[`AzureAIContentUnderstandingTool`](../tools/services/content_understanding.py),
and is included in [`AzureAIServicesToolkit`](../tools/services). See the
[Microsoft Foundry Tools guide](https://docs.langchain.com/oss/python/integrations/tools/azure_ai_services#azureaicontentunderstandingtool)
on the LangChain docs site.

## Further reading

- **Full walkthrough (notebook)** — [content_understanding_loader_demo.ipynb](../../docs/content_understanding_loader_demo.ipynb)
  covers all modalities, output modes, custom analyzers, and a RAG pipeline
  example end to end.
- **Service documentation** — [Azure Content Understanding on Microsoft Learn](https://learn.microsoft.com/azure/ai-services/content-understanding/)
- **Sample assets** — [Azure-Samples/azure-ai-content-understanding-assets](https://github.com/Azure-Samples/azure-ai-content-understanding-assets)
- **REST API reference** — [Content Understanding REST API](https://learn.microsoft.com/rest/api/contentunderstanding/)
- **LangChain integration page** — [Microsoft on docs.langchain.com](https://docs.langchain.com/oss/python/integrations/providers/microsoft)
