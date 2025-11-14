"""Sample showing embedding documents from Azure Blob Storage into Azure Search."""

import os
import warnings

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from langchain_azure_ai.vectorstores import AzureSearch
from langchain_community.document_loaders import PyPDFLoader

from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from itertools import batched


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

    for batch in batched(loader.lazy_load(), _EMBED_BATCH_SIZE):
        azure_search.add_documents(list(batch))

    print("Documents embedded and added to Azure Search index.")


if __name__ == "__main__":
    main()
