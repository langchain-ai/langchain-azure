# Ask a Kusto database in plain English

A LangChain agent that answers questions by querying
[Azure Data Explorer](https://learn.microsoft.com/azure/data-explorer/) or a
[Fabric Eventhouse](https://learn.microsoft.com/fabric/real-time-intelligence/eventhouse)
with KQL, using [`langchain-azure-kusto`](../../libs/azure-kusto).

The agent lists the tables, reads the schema of the relevant ones, writes a
KQL query and, if the query fails, reads the error and tries again. Every query
runs read-only.

## Run it

```bash
pip install langchain langchain-azure-ai langchain-azure-kusto
az login
export AZURE_AI_MODEL="azure_ai:gpt-4.1"   # plus your Foundry endpoint settings
python ask_kusto.py "Which five states had the most storm events?"
```

By default the sample queries the `Samples` database on Microsoft's public
[help cluster](https://help.kusto.windows.net/Samples), which hosts the
`StormEvents` table used throughout the KQL tutorials. To use your own data,
set `KUSTO_CLUSTER_URI` and `KUSTO_DATABASE`. The identity needs only the
Viewer role on the database.
