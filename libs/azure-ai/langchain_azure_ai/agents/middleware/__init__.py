"""Middleware for Azure AI LangChain/LangGraph agent integrations.

This module provides middleware classes for powered by Microsoft
Foundry.  Pass them via the ``middleware`` parameter of any
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

"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.agents.middleware._azure_ai_memory import (
        AzureAIMemoryMiddleware,
    )
    from langchain_azure_ai.agents.middleware.content_safety import (
        AzureContentModerationForImagesMiddleware,
        AzureContentModerationMiddleware,
        AzureGroundednessMiddleware,
        AzurePromptShieldMiddleware,
        AzureProtectedMaterialMiddleware,
        ContentSafetyViolationError,
        GroundednessInput,
        ImageModerationInput,
        PromptShieldInput,
        TextModerationInput,
        get_content_safety_annotations,
        print_content_safety_annotations,
    )

__all__ = [
    "AzureAIMemoryMiddleware",
    "AzureContentModerationMiddleware",
    "AzureContentModerationForImagesMiddleware",
    "AzureGroundednessMiddleware",
    "AzureProtectedMaterialMiddleware",
    "AzurePromptShieldMiddleware",
    "ContentSafetyViolationError",
    "GroundednessInput",
    "ImageModerationInput",
    "PromptShieldInput",
    "TextModerationInput",
    "print_content_safety_annotations",
    "get_content_safety_annotations",
]

_module_lookup = {
    "AzureAIMemoryMiddleware": "langchain_azure_ai.agents.middleware._azure_ai_memory",
    "AzureContentModerationMiddleware": (
        "langchain_azure_ai.agents.middleware.content_safety"
    ),
    "AzureContentModerationForImagesMiddleware": (
        "langchain_azure_ai.agents.middleware.content_safety"
    ),
    "AzureGroundednessMiddleware": (
        "langchain_azure_ai.agents.middleware.content_safety"
    ),
    "AzureProtectedMaterialMiddleware": (
        "langchain_azure_ai.agents.middleware.content_safety"
    ),
    "AzurePromptShieldMiddleware": (
        "langchain_azure_ai.agents.middleware.content_safety"
    ),
    "ContentSafetyViolationError": (
        "langchain_azure_ai.agents.middleware.content_safety"
    ),
    "GroundednessInput": "langchain_azure_ai.agents.middleware.content_safety",
    "ImageModerationInput": "langchain_azure_ai.agents.middleware.content_safety",
    "PromptShieldInput": "langchain_azure_ai.agents.middleware.content_safety",
    "TextModerationInput": "langchain_azure_ai.agents.middleware.content_safety",
    "print_content_safety_annotations": (
        "langchain_azure_ai.agents.middleware.content_safety"
    ),
    "get_content_safety_annotations": (
        "langchain_azure_ai.agents.middleware.content_safety"
    ),
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
