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

  - **[BREAKING CHANGE]** We raised the minimum supported Python version from 3.10 to 3.11. Users running Python 3.10 must upgrade their Python runtime before installing this release. [#1021](https://github.com/langchain-ai/langchain-azure/pull/1021)
