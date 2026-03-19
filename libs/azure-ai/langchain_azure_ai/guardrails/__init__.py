"""Guardrails for Azure AI LangChain/LangGraph integrations.

This module provides middleware classes for adding safety guardrails to any
LangGraph agent.  Pass them via the ``middleware`` parameter of
:meth:`~langchain_azure_ai.agents.v2.AgentServiceFactory.create_prompt_agent`
or any LangChain ``create_agent`` factory:

.. code-block:: python

    from langchain_azure_ai.agents.v2 import AgentServiceFactory
    from langchain_azure_ai.guardrails import (
        AzureContentSafetyMiddleware,
        AzureContentSafetyImageMiddleware,
    )

    factory = AgentServiceFactory(project_endpoint="https://my-project.api.azureml.ms/")
    agent = factory.create_prompt_agent(
        model="gpt-4.1",
        middleware=[
            AzureContentSafetyMiddleware(
                endpoint="https://my-resource.cognitiveservices.azure.com/",
                action="block",
            ),
            AzureContentSafetyImageMiddleware(
                endpoint="https://my-resource.cognitiveservices.azure.com/",
                action="block",
            ),
        ],
    )

Classes:
    AzureContentSafetyMiddleware: AgentMiddleware that screens **text** messages
        using Azure AI Content Safety.
    AzureContentSafetyImageMiddleware: AgentMiddleware that screens **image**
        content using the Azure AI Content Safety image analysis API.

Exceptions:
    ContentSafetyViolationError: Raised when content safety violations are
        detected with ``action='block'``.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.guardrails._content_safety import (
        AzureContentSafetyImageMiddleware,
        AzureContentSafetyMiddleware,
        ContentSafetyViolationError,
    )

__all__ = [
    "AzureContentSafetyMiddleware",
    "AzureContentSafetyImageMiddleware",
    "ContentSafetyViolationError",
]

_module_lookup = {
    "AzureContentSafetyMiddleware": "langchain_azure_ai.guardrails._content_safety",
    "AzureContentSafetyImageMiddleware": "langchain_azure_ai.guardrails._content_safety",
    "ContentSafetyViolationError": "langchain_azure_ai.guardrails._content_safety",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

