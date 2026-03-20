"""Middleware for Azure AI LangChain/LangGraph agent integrations.

This module provides middleware classes for adding safety guardrails to any
LangGraph agent.  Pass them via the ``middleware`` parameter of
:meth:`~langchain_azure_ai.agents.v2.AgentServiceFactory.create_prompt_agent`
or any LangChain ``create_agent`` factory:

.. code-block:: python

    from langchain_azure_ai.agents.v2 import AgentServiceFactory
    from langchain_azure_ai.agents.middleware import (
        AzureContentSafetyMiddleware,
        AzureContentSafetyImageMiddleware,
        AzureProtectedMaterialMiddleware,
        AzurePromptShieldMiddleware,
    )

    factory = AgentServiceFactory(project_endpoint="https://my-project.api.azureml.ms/")
    agent = factory.create_prompt_agent(
        model="gpt-4.1",
        middleware=[
            # Block harmful text in both input and output
            AzureContentSafetyMiddleware(
                endpoint="https://my-resource.cognitiveservices.azure.com/",
                action="block",
            ),
            # Block harmful images in user input
            AzureContentSafetyImageMiddleware(
                endpoint="https://my-resource.cognitiveservices.azure.com/",
                action="block",
            ),
            # Block protected/copyrighted content in AI output
            AzureProtectedMaterialMiddleware(
                endpoint="https://my-resource.cognitiveservices.azure.com/",
                action="block",
                apply_to_input=False,
                apply_to_output=True,
            ),
            # Block prompt injection attacks in user input and tool outputs
            AzurePromptShieldMiddleware(
                endpoint="https://my-resource.cognitiveservices.azure.com/",
                action="block",
            ),
        ],
    )

Classes:
    AzureContentSafetyMiddleware: AgentMiddleware that screens **text** messages
        using Azure AI Content Safety harm detection.
    AzureContentSafetyImageMiddleware: AgentMiddleware that screens **image**
        content using the Azure AI Content Safety image analysis API.
    AzureProtectedMaterialMiddleware: AgentMiddleware that detects protected
        (copyrighted) material in text using Azure AI Content Safety.
    AzurePromptShieldMiddleware: AgentMiddleware that detects prompt injection
        attacks (direct and indirect) using Azure AI Content Safety.

Exceptions:
    ContentSafetyViolationError: Raised when content safety violations are
        detected with ``action='block'``.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.agents.middleware._content_safety import (
        AzureContentSafetyImageMiddleware,
        AzureContentSafetyMiddleware,
        AzurePromptShieldMiddleware,
        AzureProtectedMaterialMiddleware,
        ContentSafetyViolationError,
    )

__all__ = [
    "AzureContentSafetyMiddleware",
    "AzureContentSafetyImageMiddleware",
    "AzureProtectedMaterialMiddleware",
    "AzurePromptShieldMiddleware",
    "ContentSafetyViolationError",
]

_mod = "langchain_azure_ai.agents.middleware._content_safety"
_module_lookup = {
    "AzureContentSafetyMiddleware": _mod,
    "AzureContentSafetyImageMiddleware": _mod,
    "AzureProtectedMaterialMiddleware": _mod,
    "AzurePromptShieldMiddleware": _mod,
    "ContentSafetyViolationError": _mod,
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
