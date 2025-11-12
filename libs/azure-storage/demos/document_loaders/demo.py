import os

from langchain_azure_ai.vectorstores import AzureSearch
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel


def get_azure_search():
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
    return azure_search

def create_retriever():
    azure_search = get_azure_search()
    retriever = azure_search.as_retriever(
        search_type="similarity",
        k=3,
    )
    return retriever

def test_retriever():
    retriever = create_retriever()
    query = "What is a DefaultAzureCredential?"
    documents = retriever.invoke(query)
    for i, doc in enumerate(documents):
        print(f"Result {i + 1}:\n{doc.page_content}\n")

if __name__ == "__main__":
    test_retriever()