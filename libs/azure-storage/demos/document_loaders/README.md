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
   dotenv set AZURE_STORAGE_ACCOUNT_URL "https://langchainazstoragedemo.blob.core.windows.net/documents"
   dotenv set AZURE_STORAGE_CONTAINER_NAME "your-container-name"
   dotenv set AZURE_EMBEDDING_ENDPOINT "https://saurse-ignite24-aiservice1.openai.azure.com/openai/deployments/text-embedding-3-large"
   dotenv set AZURE_CHAT_ENDPOINT "https://saurse-ignite24-aiservice1.openai.azure.com/openai/deployments/gpt-4.1-mini"
   dotenv set AZURE_AI_SEARCH_ENDPOINT "your-azure-search-service-endpoint"
   ```

3. **Create vector store** (first time only):
   This step will load documents from Azure Blob Storage and save it to the Azure AI Search vector store.
   ```bash
   python embed.py
   ```

4. **Run the agent:**
    ```bash
    python demo.py
    ```

   **Sample interaction:**
    ```text
    You: What is Azure Blob Storage?

    AI: Azure Blob Storage is a service for storing large amounts of unstructured data...
    Source:  https://<your-account-name>.blob.core.windows.net/documents/pdf_file.pdf
    ```