# `langchain-azure-kusto` (`libs/azure-kusto`)

Read-only LangChain integrations for Azure Data Explorer and Fabric Eventhouse
(both run the Kusto engine and are queried with KQL), built on
`azure-kusto-data`. `KustoDatabase` is `@beta`; every other class needs one, so
the preview warning always shows.

## Read-only guarantees

An agent writes the queries, so these are security boundaries, not style:

- Every user or agent query goes through `_execute_query` / `_aexecute_query`,
  which call `_check_read_only_query` and set the `request_readonly` request
  property. A new query path that skips either is a High-severity finding.
- Management commands are detected by the first `.` after leading whitespace and
  `//` comment lines. Never send agent text to `execute_mgmt` or to the generic
  `execute`, which routes `.`-prefixed text to the management endpoint. The only
  management command is the fixed `.show tables | project TableName`.
- Table names are validated against `[\w .-]` before they are placed inside
  `['...']`. Loosening that pattern (quotes, backslashes, `]`) opens KQL
  injection.
- `query_take_max_records` is `max_rows + 1`; the extra row is how truncation
  is reported. The loader deliberately sets no cap.

## Clients and credentials

- Sync `KustoClient` and `azure.kusto.data.aio.KustoClient` are created lazily
  and reused. The aio client owns an aiohttp session bound to its event loop,
  so a loop change must create a new client, after a best-effort close of the
  old one.
- Closing a Kusto client also closes its credential. Caller-supplied
  credentials must therefore be wrapped in `_BorrowedCredential` /
  `_BorrowedAsyncCredential`; passing one through unwrapped closes the caller's
  credential, which is a finding. Default credentials are owned and closed.
- Sync methods must reject an async credential with `ValueError`.
- Every connection string goes through `_connection_string`, which stamps
  `_set_connector_details("LangChain", <version>)` for usage attribution
  without the user name.

## Tools

- Tools set `handle_tool_error=True`. `InvalidKustoInputError` and
  `KustoServiceError` (except `KustoNetworkError`) become `ToolException` so the
  agent can fix its query; network and client errors propagate. Error text is
  capped at 2000 characters.
- `_arun` must await the async database methods, not wrap the sync ones.
- The `db` field is `exclude=True`; it must never be serialized.

## Tests

Unit tests patch `KustoClient` / `AsyncKustoClient` in
`langchain_azure_kusto.database` and must not reach the network. Coverage is
held at 100%. Integration tests skip unless `KUSTO_CLUSTER_URI` and
`KUSTO_DATABASE` are set.
