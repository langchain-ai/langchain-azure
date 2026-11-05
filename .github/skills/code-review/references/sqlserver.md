# `langchain-sqlserver` (`libs/sqlserver`)

SQL Server and Azure SQL vector store (`vectorstores.py`) and chat message
history (`chat_message_histories.py`). Built directly on SQLAlchemy with
`pyodbc`.

`SQLServer_VectorStore` is the historical public name and `SQLServer` is an
alias; both are exported and both must keep working.

## SQLAlchemy ownership

The vector store owns engine and session lifecycle. Review whether a change
leaks sessions, holds one open across an entire iteration, or closes an engine
the caller supplied — see the lifecycle section of
[azure-sdk-contracts.md](azure-sdk-contracts.md). Transaction boundaries matter
in particular: a multi-statement write that is not committed as a unit can
leave the table half-updated after a failure.

Connection string and driver handling must keep supporting both SQL
authentication and Entra ID, and must not log the connection string.

## Vector type

SQL Server's native `VECTOR` type has version and dimension requirements.

- Dimension is fixed at column creation. A change that alters the declared
  dimension, or that stops validating embedding length against it, breaks
  existing tables.
- The supported distance metrics and their score direction feed
  `_select_relevance_score_fn`. The `VectorStore` rules in
  [ecosystem-contracts.md](ecosystem-contracts.md) apply: relevance scores are
  normalized to `[0, 1]` with 1 as most similar, while the underlying SQL
  operator returns a distance. Inverting that mapping silently reverses result
  ordering, which no smoke test catches.
- Schema creation must be idempotent and must not drop or rewrite an existing
  table.

## Queries

Identifiers must be quoted through SQLAlchemy constructs and values must be
bound parameters. Table and schema names arriving from user configuration are
the usual injection vector here. Metadata filter translation must not drop
conditions on unknown operators — returning extra rows is a correctness bug,
not a leniency feature.

## Chat message history

Message serialization is persisted data: round-tripping through
`messages_from_dict` / `messages_to_dict` must preserve type, content blocks,
tool calls, and `additional_kwargs`. Session-id scoping must isolate
conversations, and ordering must be stable and explicit rather than relying on
insertion order.
