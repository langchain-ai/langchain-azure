# RAG Agent with AzureBlobStorageLoader Demo
This demo creates a RAG agent that responds to queries based on documents loaded from Azure Blob Storage.

## Quick Start
1. **Install dependencies:**
   ```bash
   python -m venv .venv
   ./.venv/Scripts/activate  # Windows only - Use `source .venv/bin/activate` on macOS/Linux
   python -m pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   dotenv set AZURE_STORAGE_ACCOUNT_URL "https://<your-account-name>.blob.core.windows.net"
   dotenv set AZURE_STORAGE_CONTAINER_NAME "your-container-name"
   dotenv set AZURE_STORAGE_BLOB_PREFIX "your-blob-prefix"  # Defaults to `None`
   dotenv set AZURE_EMBEDDING_MODEL "your-embedding-model"
   dotenv set AZURE_EMBEDDING_ENDPOINT "https://<your-azure-foundry-resource-name>.openai.azure.com/openai/deployments/<your-embedding-model>"
   dotenv set AZURE_CHAT_MODEL "your-chat-model"
   dotenv set AZURE_CHAT_ENDPOINT "https://<your-azure-foundry-resource-name>.openai.azure.com/openai/deployments/<your-chat-model>"
   dotenv set AZURE_AI_SEARCH_ENDPOINT "https://<your-azure-search-resource-name>.search.windows.net"
   dotenv set AZURE_AI_SEARCH_INDEX_NAME "your-index-name"  # Defaults to `demo-documents`
   ```

3. **Create vector store** (first time only):

   This step will list blobs as documents from an Azure Blob Storage container and save it to the Azure AI Search vector store. To specify which blobs to return, set the `AZURE_STORAGE_BLOB_PREFIX` environment variable, otherwise all blobs in the container will be returned.
   ```bash
   python embed.py
   ```

4. **Run the agent:**

   This step runs the chatbot agent which uses the context saved to the Azure AI Search vector store to respond to questions.
   ```bash
   python query.py
   ```

   **Sample interaction:**
   ```text
   You: What is Azure Blob Storage?

   AI: Azure Blob Storage is a service for storing large amounts of unstructured data...
   Source:  https://<your-account-name>.blob.core.windows.net/<your-container-name>/pdf_file.pdf
   ```