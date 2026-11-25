"""LangChain and LangGraph integrations for Azure compute and durable execution.

This package covers three capabilities, each in its own subpackage behind its
own optional extra:

- ``dynamic_sessions`` -- managed, ephemeral code-interpreter sessions
  (``Microsoft.App/sessionPools``). Install with
  ``pip install "langchain-azure-compute[dynamic-sessions]"``.
- ``sandboxes`` -- programmable, stateful isolated compute with snapshots,
  volumes, and egress policies (``Microsoft.App/sandboxGroups``). Install with
  ``pip install "langchain-azure-compute[sandboxes]"``.
- ``graph`` -- durable execution for compiled LangGraph graphs and LangChain
  agents on Azure Durable Task Scheduler (``Microsoft.DurableTask/schedulers``).
  Install with ``pip install "langchain-azure-compute[durabletask]"``.

See the README for guidance on each subpackage.
"""

__all__: list[str] = []
