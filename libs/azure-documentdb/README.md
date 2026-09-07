# langchain-azure-documentdb

Azure DocumentDB (with MongoDB compatibility) integrations for
[LangChain](https://python.langchain.com/).

## Installation

```bash
pip install langchain-azure-documentdb
```

## Integrations

| Integration | Class | Description |
|---|---|---|
| **Vector Store** | `AzureDocumentDBVectorSearch` | Vector search for Azure DocumentDB clusters |

## Usage

```python
from typing import Optional

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from langchain_azure_documentdb import AzureDocumentDBVectorSearch
from pymongo import MongoClient
from pymongo.auth_oidc import OIDCCallback, OIDCCallbackContext, OIDCCallbackResult


class AzureIdentityTokenCallback(OIDCCallback):
    def __init__(self, credential: TokenCredential) -> None:
        self.credential = credential

    def fetch(
        self, context: OIDCCallbackContext
    ) -> Optional[OIDCCallbackResult]:
        token = self.credential.get_token(
            "https://ossrdbms-aad.database.windows.net/.default"
        )
        return OIDCCallbackResult(access_token=token.token)


credential = DefaultAzureCredential()
client = MongoClient(
    "mongodb+srv://<cluster-name>.global.mongocluster.cosmos.azure.com/",
    authMechanism="MONGODB-OIDC",
    authMechanismProperties={
        "OIDC_CALLBACK": AzureIdentityTokenCallback(credential)
    },
    retryWrites=False,
    tls=True,
)
collection = client["my-database"]["my-collection"]

vectorstore = AzureDocumentDBVectorSearch(
    collection=collection,
    embedding=embedding,
    index_name="vectorSearchIndex",
)

vectorstore.add_texts(["Azure DocumentDB supports vector search."])
results = vectorstore.similarity_search("What does DocumentDB support?", k=3)
```

## End-to-end tests

The integration test uses Microsoft Entra ID for both Azure DocumentDB and
Azure OpenAI. Authenticate locally with Azure CLI, then set the resource
configuration and run the test:

```powershell
az login --tenant <tenant-id>
az account set --subscription <subscription-id>

$env:AZURE_TOKEN_CREDENTIALS = "AzureCliCredential"
$env:AZURE_DOCUMENTDB_CLUSTER_NAME = "<cluster-name>"
$env:AZURE_OPENAI_ENDPOINT = "https://<resource-name>.openai.azure.com"
$env:OPENAI_EMBEDDINGS_DEPLOYMENT = "text-embedding-3-small"
$env:AZURE_OPENAI_API_VERSION = "2023-05-15"

pytest tests/integration_tests/test_vectorstore.py -v
```

The test creates a uniquely named temporary database and removes it after the
test, including when an assertion or service call fails.

For connection and Microsoft Entra ID authentication guidance, see the
[Azure DocumentDB documentation](https://learn.microsoft.com/azure/documentdb/).