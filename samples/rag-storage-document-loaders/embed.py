"""Sample showing embedding documents from Azure Blob Storage into Azure Search."""

import os
import warnings


from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from langchain_azure_ai.vectorstores import AzureSearch
from langchain_community.document_loaders import PyPDFLoader

from langchain_azure_storage.document_loaders import AzureBlobStorageLoader


load_dotenv()
warnings.filterwarnings("ignore", message=".*is currently in preview.*")
warnings.filterwarnings("ignore", message=".*`AzureBlobStorageLoader` is in public preview.*")


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
        credential=DefaultAzureCredential(),
        model="text-embedding-3-large",
        client_kwargs={
            "credential_scopes": ["https://cognitiveservices.azure.com/.default"]
        },
    )

    azure_search = AzureSearch(
        azure_search_endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
        azure_search_key=None,
        azure_credential=DefaultAzureCredential(),
        additional_search_client_options={
            "credential_scopes": ["https://cognitiveservices.azure.com/.default"]
        },
        index_name="demo-documents",
        embedding_function=embed_model,
    )

    batch = []
    batch_size = 50
    for doc in loader.lazy_load():
        batch.append(doc)
        if len(batch) == batch_size:
            azure_search.add_documents(batch)
            batch = []
    if batch:
        azure_search.add_documents(batch)

    print("Documents embedded and added to Azure Search index.")


if __name__ == "__main__":
    main()
