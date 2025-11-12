import os

from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_azure_ai.vectorstores import AzureSearch


def main():
    embed_model = AzureAIEmbeddingsModel(
        endpoint=os.environ["AZURE_EMBEDDING_ENDPOINT"],
        credential=os.environ["AZURE_AI_CREDENTIAL"],
        model="text-embedding-3-large",
    )

    azure_search = AzureSearch(
        azure_search_endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
        azure_search_key=os.environ["AZURE_AI_SEARCH_KEY"],
        index_name="demo-documents",
        embedding_function=embed_model,
    )

    loader = AzureBlobStorageLoader(
        account_url=os.environ["AZURE_STORAGE_ACCOUNT_URL"],
        container_name=os.environ["AZURE_STORAGE_CONTAINER_NAME"],
        blob_names=["pdf_test.pdf", "pdf_file2.pdf", "pdf_file3.pdf", "pdf_file4.pdf", "pdf_file5.pdf"],
        loader_factory=PyPDFLoader,
    )

    for doc in loader.lazy_load():
        azure_search.add_documents([doc])

    print("Documents embedded and added to Azure Search index.")


if __name__ == "__main__":
    main()