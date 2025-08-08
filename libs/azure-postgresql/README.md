# langchain-azure-postgresql

`langchain-azure-postgresql` is a Python package that implements both asynchronous
and synchronous `VectorStore` support for Azure Database for PostgreSQL. Specifically,
this package adds support for

1. Microsoft Entra ID (formerly Azure AD) authentication when connecting to your
   Azure Database for PostgreSQL instances, and,
1. DiskANN indexing algorithm when indexing your (semantic) vectors.

This way, you can leverage your Azure Database for PostgreSQL instances as secure
and fast vector stores for your LangChain workflows.

> [!NOTE]
> `langchain-azure-postgresql` currently supports Python 3.10 and above.

## Installation

To install `langchain-azure-postgresql`, you need to install the necessary Python
packages:

```cmd
$ python3 -m pip install langchain langchain-azure-postgresql langchain-openai
# logs stripped for brevity
```

## Usage

Once the packages are installed, you can use Azure Database for PostgreSQL instances
as vector stores:

```python
import os

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_azure_postgresql.common import AzurePGConnectionPool, ConnectionInfo
from langchain_azure_postgresql.langchain import AzurePGVectorStore

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

host = os.getenv("PGHOST", "localhost")
connection_pool = AzurePGConnectionPool(azure_conn_info=ConnectionInfo(host=host))

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = AzurePGVectorStore(connection_pool=connection_pool, embedding=embedding)

vector_store.add_documents(documents)
```

The code snippet above will try to connect to your PostgreSQL `host`, as defined
by the environment variable `PGHOST`, and fall back to connecting to `localhost`
if the environment variable is not defined. By default, `ConnectionInfo` objects
try to use Microsoft Entra ID to login to the PostgreSQL instances/hosts.

Please see the documentation for more details on configuring various classes
provided by `langchain-azure-postgresql`.

## Development

The development environment for this package is managed by [`uv`][uv-link], an
up-and-coming and versatile Python package and project manager.

To create the development environment, you first need to install `uv` in your
development environment (unless you are using the development container setup
as provided by this repository). Once `uv` is properly installed and set up,
you can run the following commands to synchronize and activate the development
environment:

```cmd
$ uv sync --all-extras # synchronize the development environment from uv.lock
# logs stripped for brevity
$ source .venv/bin/activate # or, as appropriate for your shell, e.g., fish
```

Once the development environment is synchronized and activated, you can start
contributing changes to the package. For test automation, the package leverages
[`tox`][tox-link], a test automation and standardization framework in Python.

There are some pre-defined test environments and labels managed by `tox`:

```cmd
$ tox list
default environments:
lint    -> Run lint checks on the code base
package -> Run packaging checks on the code base
type    -> Run type checks on the code base
3.10    -> Run tests under Python 3.10
3.11    -> Run tests under Python 3.11
3.12    -> Run tests under Python 3.12
3.13    -> Run tests under Python 3.13
```

The default environments are for running lint checks, packaging checks, type
checks, and end-to-end tests for different Python versions, respectively. You
can selectively run an environment via, e.g., `tox run -e lint`, or, otherwise,
run the full suite of tests via `tox`. There is a special label called `test`,
which will run _only_ the end-to-end tests for all the supported Python versions
and can be run via the following:

```cmd
$ PGHOST=<host>.postgres.database.azure.com PGPASSWORD='your_password' tox run -m test
# logs stripped for brevity
```

For more information on the supported test flags and environment variables, please
check the output of `pytest --help`.

[uv-link]: https://docs.astral.sh/uv/
[tox-link]: https://tox.wiki/
