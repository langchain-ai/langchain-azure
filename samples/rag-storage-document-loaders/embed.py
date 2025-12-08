"""Sample showing embedding documents from Azure Blob Storage into Azure Search."""

import os
import warnings

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from langchain_azure_ai.vectorstores import AzureSearch
from langchain_community.document_loaders import PyPDFLoader

from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger("pypdf")
logger.setLevel(logging.ERROR)

load_dotenv()
warnings.filterwarnings("ignore", message=".*preview.*")

_CREDENTIAL = DefaultAzureCredential()
_COGNITIVE_CREDENTIAL_SCOPES = {
    "credential_scopes": ["https://cognitiveservices.azure.com/.default"]
}
_EMBED_BATCH_SIZE = 50


def main() -> None:
    """Embed documents from Azure Blob Storage into Azure Search."""
    loader = AzureBlobStorageLoader(
        account_url=os.environ["AZURE_STORAGE_ACCOUNT_URL"],
        container_name=os.environ["AZURE_STORAGE_CONTAINER_NAME"],
        prefix=os.environ.get("AZURE_STORAGE_BLOB_PREFIX", None),
        loader_factory=PyPDFLoader,  # Parses blobs as PDFs into LangChain Documents
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=12000,
        chunk_overlap=500,
    )

    embed_model = AzureAIEmbeddingsModel(
        endpoint=os.environ["AZURE_EMBEDDING_ENDPOINT"],
        credential=_CREDENTIAL,
        model=os.environ["AZURE_EMBEDDING_MODEL"],
        client_kwargs=_COGNITIVE_CREDENTIAL_SCOPES.copy(),
    )

    azure_search = AzureSearch(
        azure_search_endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
        azure_search_key=None,
        azure_credential=_CREDENTIAL,
        additional_search_client_options=_COGNITIVE_CREDENTIAL_SCOPES,
        index_name=os.environ.get("AZURE_AI_SEARCH_INDEX_NAME", "demo-documents"),
        embedding_function=embed_model,
    )

    docs = []
    total_processed = 0
    blobs_seen = set()
    blob_progress = get_progress_bar()
    for doc in loader.lazy_load():
        update_progress_bar(doc, blobs_seen, blob_progress)
        splits = text_splitter.split_documents([doc])
        docs.extend(splits)
        if len(docs) >= _EMBED_BATCH_SIZE:
            azure_search.add_documents(docs)
            total_processed += len(docs)
            docs = []

    if docs:
        azure_search.add_documents(docs)
        total_processed += len(docs)

    blob_progress.close()
    print(
        f"Complete: {total_processed} documents across {len(blobs_seen)} blobs embedded and added to Azure Search index."
    )


def get_progress_bar() -> tqdm:
    blob_service_client = BlobServiceClient(
        account_url=os.environ["AZURE_STORAGE_ACCOUNT_URL"],
        credential=_CREDENTIAL,
    )
    container_client = blob_service_client.get_container_client(
        os.environ["AZURE_STORAGE_CONTAINER_NAME"]
    )
    prefix = os.environ.get("AZURE_STORAGE_BLOB_PREFIX", None)
    blob_list = list(container_client.list_blobs(name_starts_with=prefix))

    blob_progress = tqdm(total=len(blob_list), desc="Processing blobs", unit=" blobs")
    return blob_progress


def update_progress_bar(doc, blobs_seen, blob_progress) -> None:
    blob = doc.metadata.get("source")
    if blob not in blobs_seen:
        blob_progress.update(1)
    blobs_seen.add(blob)


if __name__ == "__main__":
    main()
