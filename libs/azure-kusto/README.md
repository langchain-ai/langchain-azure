# langchain-azure-kusto

LangChain integrations for [Azure Data Explorer](https://learn.microsoft.com/azure/data-explorer/)
and [Eventhouse in Microsoft Fabric](https://learn.microsoft.com/fabric/real-time-intelligence/eventhouse).

Both services run the Kusto engine and are queried with
[KQL (Kusto Query Language)](https://learn.microsoft.com/kusto/query/), so one
package covers both:

| Class | What it does |
|---|---|
| `KustoDatabase` | A read-only connection to one database |
| `KustoToolkit` | Tools that let an agent list tables, read their schema and run KQL queries |
| `KustoLoader` | Turns the rows of a KQL query into LangChain `Document`s |

All classes are in public preview (`@beta`); the API may change.

## Installation

```bash
pip install langchain-azure-kusto
```

## Connect to a database

```python
from langchain_azure_kusto import KustoDatabase

db = KustoDatabase(
    cluster_uri="https://help.kusto.windows.net",
    database="Samples",
)
print(db.run("StormEvents | summarize count() by State | top 3 by count_"))
```

`cluster_uri` is the cluster's or Eventhouse's **query URI**:

- Azure Data Explorer: `https://<cluster>.<region>.kusto.windows.net`
- Fabric Eventhouse: `https://<id>.<region>.kusto.fabric.microsoft.com`
  (copy it from the Eventhouse details page)

Results are returned as CSV text, at most `max_rows` rows per result
(default 100). A note is appended when rows were left out.

### Authentication

By default `DefaultAzureCredential` is used, so `az login`, a managed identity
or environment variables all work. Pass `credential=` to use another
`azure-identity` credential:

```python
from azure.identity import ManagedIdentityCredential

db = KustoDatabase(
    "https://<cluster>.<region>.kusto.windows.net",
    "<database>",
    credential=ManagedIdentityCredential(),
)
```

A synchronous credential works with both the sync and async methods. An
async credential (from `azure.identity.aio`) works with the async methods only.

### Closing

The Kusto clients are created on first use and reused. Close them when you are
done, or use a context manager. A credential you pass in is never closed.

```python
with KustoDatabase(cluster_uri, database) as db:
    ...

async with KustoDatabase(cluster_uri, database) as db:
    ...
```

## Let an agent query the database

```python
from langchain.agents import create_agent

from langchain_azure_kusto import KustoDatabase, KustoToolkit

db = KustoDatabase("https://help.kusto.windows.net", "Samples")
agent = create_agent(
    model="<your-model>",
    tools=KustoToolkit(db=db).get_tools(),
    system_prompt="Answer questions by querying the Kusto database with KQL.",
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Which state had the most storms?"}]}
)
```

The toolkit returns three tools:

| Tool | Input | Returns |
|---|---|---|
| `kusto_list_tables` | none | The table names |
| `kusto_table_schema` | `table_names`, comma-separated | Column names and types, plus a few sample rows |
| `kusto_query` | `query` | The query results as CSV |

Query errors (bad syntax, unknown table or column) are returned to the agent
as text so it can fix the query and retry. Network errors are raised.

A runnable version is in [samples/kusto-agent](../../samples/kusto-agent).

## Load query results as documents

```python
from langchain_azure_kusto import KustoDatabase, KustoLoader

db = KustoDatabase("https://help.kusto.windows.net", "Samples")
loader = KustoLoader(
    "StormEvents | where isnotempty(EpisodeNarrative) | take 100",
    db,
    page_content_columns=["EpisodeNarrative"],
    metadata_columns=["State", "EventType", "StartTime"],
)
docs = loader.load()
```

Each row becomes one `Document`. The page content is one `column: value` line
per content column; by default every column that is not a metadata column.
All rows are fetched in one request, so limit the result with `take` or a
filter.

## Security

These classes are meant for reading data, and an agent decides which queries
run. To limit what it can do:

- **Grant only the Viewer role.** Use an identity with the *Database Viewer*
  role in Azure Data Explorer, or the *Viewer* role on the Eventhouse or
  workspace in Fabric. See
  [role-based access control](https://learn.microsoft.com/kusto/access-control/role-based-access-control).
- Every query runs with the
  [`request_readonly`](https://learn.microsoft.com/kusto/api/rest/request-properties)
  request property, so it cannot change data.
- Management commands (text starting with `.`) are rejected before they are
  sent. The table-list tool runs one fixed command, `.show tables`.
- `max_rows` and `query_timeout` cap how much data and time a single query
  can use.

These checks reduce risk; they do not replace least-privilege access.

## Telemetry

Requests identify themselves to the service as coming from the LangChain
connector and this package version, so the Kusto team can see how the
connector is used. No user name is sent.

## Development

See [CONTRIBUTING.md](../../CONTRIBUTING.md) in the repository root.

```bash
make test          # unit tests
make lint          # ruff + mypy
make check_imports # import smoke check
```

### Running the integration tests

They run against a live database and skip themselves when the variables below
are unset, so `make integration_tests` is safe to run with nothing configured.

```bash
cp .env.example .env   # then fill in your database
az login
make integration_tests
```

| Variable | Meaning |
|---|---|
| `KUSTO_CLUSTER_URI` | Query URI of the cluster or Eventhouse |
| `KUSTO_DATABASE` | Database name; it must contain at least one table |

The identity needs the Viewer role on the database.

## Changelog

- **0.1.0**:

  - Initial release with `KustoDatabase`, `KustoToolkit` (`KustoListTablesTool`, `KustoTableSchemaTool`, `KustoQueryTool`) and `KustoLoader` for Azure Data Explorer and Fabric Eventhouse.
