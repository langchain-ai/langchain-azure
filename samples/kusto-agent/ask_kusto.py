"""Ask questions about a Kusto database in plain English.

A LangChain agent uses the Kusto toolkit to list tables, read their schema and
run read-only KQL queries against Azure Data Explorer or a Fabric Eventhouse.

Requires:
    pip install langchain langchain-azure-ai langchain-azure-kusto

Environment variables:
    AZURE_AI_MODEL     - chat model, for example azure_ai:gpt-4.1
    KUSTO_CLUSTER_URI  - defaults to Microsoft's public help cluster,
                         https://help.kusto.windows.net
    KUSTO_DATABASE     - defaults to its Samples database

Authentication uses DefaultAzureCredential (for example, `az login`). The
signed-in identity needs only the Viewer role on the database.
"""

import os
import sys

from langchain.agents import create_agent

from langchain_azure_kusto import KustoDatabase, KustoToolkit

SYSTEM_PROMPT = """You answer questions by querying a Kusto database with KQL.
First list the tables, then read the schema of the relevant ones, then query.
Aggregate in the query rather than reading raw rows. Say which query you ran."""

DEFAULT_QUESTION = "Which five states had the most storm events, and how many?"


def main() -> None:
    """Run the agent on one question from the command line."""
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    with KustoDatabase(
        cluster_uri=os.environ.get(
            "KUSTO_CLUSTER_URI", "https://help.kusto.windows.net"
        ),
        database=os.environ.get("KUSTO_DATABASE", "Samples"),
        max_rows=50,
    ) as db:
        agent = create_agent(
            model=os.environ["AZURE_AI_MODEL"],
            tools=KustoToolkit(db=db).get_tools(),
            system_prompt=SYSTEM_PROMPT,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
