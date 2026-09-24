# langchain-sqlserver

This package contains the LangChain integration for Azure SQL and SQL Server to take advantage of the newly introduced [Vector data type](https://learn.microsoft.com/sql/t-sql/data-types/vector-data-type?view=azuresqldb-current&tabs=python).

## Installation

```bash
pip install -U langchain-sqlserver
```

## Samples

Samples on how to use the `langchain-sqlserver` package with SQL Server and Azure SQL are available here: https://github.com/Azure-Samples/azure-sql-langchain

## Changelog

- **1.0.2**:

  - **[NEW]** We added `SQLServerChatMessageHistory` for persisting chat messages in SQL Server and Azure SQL. [#629](https://github.com/langchain-ai/langchain-azure/pull/629)
  - **[NEW]** We added an `upsert` option to vector-store text and document insertion methods. [#628](https://github.com/langchain-ai/langchain-azure/pull/628)
  - We introduced `SQLServerVectorStore` as the canonical vector-store class name while retaining `SQLServer_VectorStore` as a deprecated alias. [#798](https://github.com/langchain-ai/langchain-azure/pull/798)
  - We added an opt-in binary collation for vector-store `custom_id` columns. [#801](https://github.com/langchain-ai/langchain-azure/pull/801)
  - We fixed `$nin` metadata filters so they compile and execute correctly with SQLAlchemy. [#853](https://github.com/langchain-ai/langchain-azure/pull/853)
  - We fixed the package dependencies to require `SQLAlchemy[asyncio]`, so the `greenlet` runtime dependency needed by the async vector store is always installed. [#1083](https://github.com/langchain-ai/langchain-azure/pull/1083)
  - We raised the minimum supported Python version from 3.10 to 3.11 in accordance with the repository's Python support policy. Users running Python 3.10 must upgrade their runtime to install this release. [#1021](https://github.com/langchain-ai/langchain-azure/pull/1021)
