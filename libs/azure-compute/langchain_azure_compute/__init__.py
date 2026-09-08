"""LangChain integrations for Azure Container Apps.

Azure Container Apps offers several compute types. This package covers the two
that run agent-authored code, each in its own subpackage behind its own extra:

- ``dynamic_sessions`` -- managed, ephemeral code-interpreter sessions
  (``Microsoft.App/sessionPools``). Install with
  ``pip install "langchain-azure-compute[dynamic-sessions]"``.
- ``sandboxes`` -- programmable, stateful isolated compute with snapshots,
  volumes, and egress policies (``Microsoft.App/sandboxGroups``). Install with
  ``pip install "langchain-azure-compute[sandboxes]"``.

The two are distinct Azure products, not tiers of one: they use different ARM
resource types and different data planes. See the README for guidance on
choosing between them.
"""

__all__: list[str] = []
