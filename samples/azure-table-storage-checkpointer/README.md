# LangGraph Checkpointer with Azure Table Storage
This demo persists a LangGraph graph's state in Azure Table Storage using `AzureTableStorageSaver`. It runs a small graph twice on the same thread, shows the second run picking up the first run's state, then inspects the saved state and history. No LLM is required.

## Quick Start

1. **Install dependencies** (from this directory):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: ./.venv/Scripts/activate
   python -m pip install -r requirements.txt
   ```

2. **Choose a Table Storage target and set one environment variable:**

   - **Azurite emulator** (no Azure account needed). Start the Table service, then use its connection string:
     ```bash
     npx azurite-table --silent --location .azurite &
     export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"
     ```
   - **Live storage account.** Authenticates with [`DefaultAzureCredential`](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) (e.g. [`Azure CLI`](https://learn.microsoft.com/en-us/cli/azure/authenticate-azure-cli?view=azure-cli-latest)), which needs the `Storage Table Data Contributor` role on the account:
     ```bash
     export AZURE_STORAGE_TABLE_ENDPOINT="https://<your-account-name>.table.core.windows.net"
     ```

3. **Run the sample:**
   ```bash
   python checkpointer.py
   ```

   The sample creates a uniquely named table and deletes it when it finishes.

   **Sample output:**
   ```text
   === Turn 1 ===
     history: ['Alice: hi', 'node saw 1 entries']

   === Turn 2 (state is restored from Table Storage) ===
     history: ['Alice: hi', 'node saw 1 entries', 'Alice: again', 'node saw 3 entries']
   ```
