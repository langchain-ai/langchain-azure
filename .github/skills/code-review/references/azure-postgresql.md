# `langchain-azure-postgresql` (`libs/azure-postgresql`)

Azure Database for PostgreSQL integration: connection management
(`common/`, `common/aio/`) and vector store (`langchain/`, `langchain/aio/`),
under `src/`. Built on `psycopg` and `pgvector`.

Its `.github/copilot-instructions.md` asks for Sphinx-style docstrings, but
`pyproject.toml` configures pydocstyle with `convention = "google"`. Follow the
tooling. This package also uses `uv` and `tox` rather than the repository-wide
`make` targets — check its `Makefile` and `tox.ini` before claiming a command
is wrong.

## Connections, pools, and credentials

- Entra ID tokens **expire**. Any caching of a token or of a connection built
  from one must handle refresh; a pool that outlives the token it was created
  with will start failing mid-process. Password and Entra ID paths must both
  keep working.
- Pool ownership follows the rule in
  [azure-sdk-contracts.md](azure-sdk-contracts.md): a caller-supplied pool or
  connection must never be closed by the vector store, and one the store
  created must be released. Getting this backwards leaks connections under
  load or closes a pool the application is still using.
- Sync and async connection code are separate implementations; changes must be
  mirrored unless there is a stated reason not to.
- SSL and network configuration defaults are security-relevant: weakening a
  default is a High-severity finding.

## SQL construction

Every identifier — table, schema, column, index name — must be quoted through
`psycopg.sql.Identifier`, and every value must be a bound parameter. String
interpolation of a user-controlled value into SQL is a High-severity injection
finding with no exceptions. This applies to the metadata filter translator as
much as to the main query path: filter keys arrive from user data and become
identifiers or JSON paths.

Schema and table creation must stay idempotent, must not silently drop or
rewrite existing data, and must remain compatible with tables created by
earlier versions of the package.

## Vector search

- The `VectorStore` contract in
  [ecosystem-contracts.md](ecosystem-contracts.md) applies, including
  relevance scores normalized to `[0, 1]` with 1 as most similar. `pgvector`
  distance operators return *distances*, so the conversion direction is the
  usual source of bugs here.
- Index type (DiskANN, HNSW, IVFFlat) and its build parameters must match the
  distance operator used at query time, or the index is silently not used.
  A query change that stops matching the index is a performance regression
  worth flagging even though results stay correct.
- Extension requirements (`vector`, `pg_diskann`, `azure_ai`) must be checked
  or documented rather than assumed.
- Embedding dimension mismatches must fail with a clear error, not a database
  error from three layers down.

## Filters

The filter-to-SQL translator is the highest-risk code in the package: it must
reject or safely escape unknown operators rather than pass them through, must
produce the same semantics in sync and async paths, and must handle `None`,
empty dicts, and nested boolean combinations without silently dropping a
condition. A dropped condition returns *more* rows than the caller asked for,
which no test that only checks "results are non-empty" will catch.
