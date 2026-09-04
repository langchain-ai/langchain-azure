"""End-to-end tests for Azure DocumentDB vector search."""

import os
import time
import uuid
from typing import Optional

import pytest
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureOpenAIEmbeddings
from pymongo import MongoClient
from pymongo.auth_oidc import OIDCCallback, OIDCCallbackContext, OIDCCallbackResult

from langchain_azure_documentdb import (
    AzureDocumentDBSimilarityType,
    AzureDocumentDBVectorSearch,
    AzureDocumentDBVectorSearchType,
)

DOCUMENTDB_TOKEN_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"
OPENAI_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


class AzureIdentityTokenCallback(OIDCCallback):
    """Provide Microsoft Entra ID tokens to PyMongo."""

    def __init__(self, credential: TokenCredential) -> None:
        self.credential = credential

    def fetch(
        self, context: OIDCCallbackContext
    ) -> Optional[OIDCCallbackResult]:
        token = self.credential.get_token(DOCUMENTDB_TOKEN_SCOPE)
        return OIDCCallbackResult(access_token=token.token)


@pytest.mark.requires("azure-documentdb", "azure-openai")
@pytest.mark.skipif(
    not os.getenv("AZURE_DOCUMENTDB_CLUSTER_NAME")
    or not os.getenv("AZURE_OPENAI_ENDPOINT"),
    reason="Azure DocumentDB and Azure OpenAI E2E settings are not configured",
)
def test_vector_search_with_entra_id() -> None:
    """Insert, index, and query documents using Microsoft Entra ID."""
    cluster_name = os.environ["AZURE_DOCUMENTDB_CLUSTER_NAME"]
    openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    embedding_deployment = os.getenv(
        "OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-small"
    )
    openai_api_version = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2023-05-15"
    )
    database_name = f"langchain_e2e_{uuid.uuid4().hex[:12]}"
    collection_name = "documents"
    index_name = "vectorSearchIndex"

    credential = DefaultAzureCredential()
    client = MongoClient(
        f"mongodb+srv://{cluster_name}.global.mongocluster.cosmos.azure.com/",
        appname="langchain-azure-documentdb-e2e",
        authMechanism="MONGODB-OIDC",
        authMechanismProperties={
            "OIDC_CALLBACK": AzureIdentityTokenCallback(credential)
        },
        connectTimeoutMS=30_000,
        maxIdleTimeMS=120_000,
        retryWrites=False,
        serverSelectionTimeoutMS=30_000,
        tls=True,
    )

    try:
        client.admin.command("ping")
        collection = client[database_name][collection_name]
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=openai_endpoint,
            azure_deployment=embedding_deployment,
            openai_api_version=openai_api_version,
            azure_ad_token_provider=get_bearer_token_provider(
                credential, OPENAI_TOKEN_SCOPE
            ),
        )
        dimension = len(embeddings.embed_query("dimension probe"))
        vectorstore = AzureDocumentDBVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=index_name,
        )
        vectorstore.add_texts(
            [
                "Azure DocumentDB supports MongoDB-compatible vector search.",
                "A sandwich is food placed between slices of bread.",
                "The Pacific Ocean is the largest ocean on Earth.",
            ]
        )
        vectorstore.create_index(
            num_lists=1,
            dimensions=dimension,
            similarity=AzureDocumentDBSimilarityType.COS,
            kind=AzureDocumentDBVectorSearchType.VECTOR_IVF,
        )

        deadline = time.monotonic() + 60
        results = []
        while time.monotonic() < deadline:
            results = vectorstore.similarity_search(
                "What food uses two slices of bread?",
                k=1,
                kind=AzureDocumentDBVectorSearchType.VECTOR_IVF,
            )
            if results:
                break
            time.sleep(2)

        assert results
        assert "sandwich" in results[0].page_content.lower()
    finally:
        client.drop_database(database_name)
        client.close()
        credential.close()