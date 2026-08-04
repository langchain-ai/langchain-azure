"""LangChain integrations for Azure Container Apps.

Azure Container Apps offers several compute types. This package covers the ones
that run agent-authored code, each in its own subpackage behind its own extra:

- ``dynamic_sessions`` -- managed, ephemeral code-interpreter sessions
  (``Microsoft.App/sessionPools``). Install with
  ``pip install "langchain-azure-container-apps[dynamic-sessions]"``.

"""

__all__: list[str] = []
