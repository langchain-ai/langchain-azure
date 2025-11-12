"""Sample showing embedding documents from Azure Blob Storage into Azure Search."""

import os
import warnings

from azure.identity import DefaultAzureCredential
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from langchain_azure_ai.vectorstores import AzureSearch
from langchain_community.document_loaders import PyPDFLoader

from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

warnings.filterwarnings("ignore")


def main():
    """Embed documents from Azure Blob Storage into Azure Search."""
    loader = AzureBlobStorageLoader(
        account_url=os.environ["AZURE_STORAGE_ACCOUNT_URL"],
        container_name=os.environ["AZURE_STORAGE_CONTAINER_NAME"],
        blob_names=[
            "pdf_test.pdf",
            "pdf_file2.pdf",
            "pdf_file3.pdf",
            "pdf_file4.pdf",
            "pdf_file5.pdf",
        ],
        loader_factory=PyPDFLoader,
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

    for doc in loader.lazy_load():
        azure_search.add_documents([doc])

    print("Documents embedded and added to Azure Search index.")


if __name__ == "__main__":
    main()
