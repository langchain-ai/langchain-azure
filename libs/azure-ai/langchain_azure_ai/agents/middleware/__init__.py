"""Middleware for Azure AI LangChain/LangGraph agent integrations.

This module provides middleware classes for adding safety guardrails to any
LangGraph agent.  Pass them via the ``middleware`` parameter of any
LangChain ``create_agent`` factory:

.. code-block:: python

    from langchain.agents import create_agent
    from langchain_azure_ai.agents.middleware import (
        AzureContentModerationMiddleware,
        AzureContentModerationForImagesMiddleware,
        AzureGroundednessMiddleware,
        AzureProtectedMaterialMiddleware,
        AzurePromptShieldMiddleware,
    )

    agent = create_agent(
        model="azure_ai:gpt-4.1",
        middleware=[
            # Block harmful text in both input and output
            AzureContentModerationMiddleware(
                exit_behavior="error",
            ),
            # Block harmful images in user input
            AzureContentModerationForImagesMiddleware(
                exit_behavior="error",
            ),
            # Block protected/copyrighted content in AI output
            AzureProtectedMaterialMiddleware(
                exit_behavior="continue",
                apply_to_input=False,
                apply_to_output=True,
            ),
            # Block prompt injection attacks in user input and tool outputs
            AzurePromptShieldMiddleware(
                exit_behavior="error",
            ),
        ],
    )

Classes:
    AzureContentModerationMiddleware: AgentMiddleware that screens **text** messages
        using Azure AI Content Safety harm detection.
    AzureContentModerationImageMiddleware: AgentMiddleware that screens **image**
        content using the Azure AI Content Safety image analysis API.
    AzureProtectedMaterialMiddleware: AgentMiddleware that detects protected
        (copyrighted) material in text using Azure AI Content Safety.
    AzurePromptShieldMiddleware: AgentMiddleware that detects prompt injection
        attacks (direct and indirect) using Azure AI Content Safety.
    AzureGroundednessMiddleware: AgentMiddleware that evaluates groundedness
        of model outputs and annotates the state with evaluation results.

"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.agents.middleware.content_safety import (
        AzureContentModerationForImagesMiddleware,
        AzureContentModerationMiddleware,
        AzureGroundednessMiddleware,
        AzurePromptShieldMiddleware,
        AzureProtectedMaterialMiddleware,
        ContentSafetyViolationError,
        GroundednessInput,
        get_content_safety_annotations,
        print_content_safety_annotations,
    )

__all__ = [
    "AzureContentModerationMiddleware",
    "AzureContentModerationForImagesMiddleware",
    "AzureGroundednessMiddleware",
    "AzureProtectedMaterialMiddleware",
    "AzurePromptShieldMiddleware",
    "ContentSafetyViolationError",
    "GroundednessInput",
    "print_content_safety_annotations",
    "get_content_safety_annotations",
]

_mod = "langchain_azure_ai.agents.middleware.content_safety"
_module_lookup = {
    "AzureContentModerationMiddleware": _mod,
    "AzureContentModerationForImagesMiddleware": _mod,
    "AzureGroundednessMiddleware": _mod,
    "AzureProtectedMaterialMiddleware": _mod,
    "AzurePromptShieldMiddleware": _mod,
    "ContentSafetyViolationError": _mod,
    "GroundednessInput": _mod,
    "print_content_safety_annotations": _mod,
    "get_content_safety_annotations": _mod,
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
