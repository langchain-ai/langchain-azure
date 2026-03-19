"""Backward-compatibility shim.

The implementation has moved to
:mod:`langchain_azure_ai.agents.middleware._content_safety`.
"""

from langchain_azure_ai.agents.middleware._content_safety import (  # noqa: F401
    AzureContentSafetyImageMiddleware,
    AzureContentSafetyMiddleware,
    AzurePromptShieldMiddleware,
    AzureProtectedMaterialMiddleware,
    ContentSafetyViolationError,
    _AzureContentSafetyBaseMiddleware,
    _ContentSafetyState,
)

__all__ = [
    "AzureContentSafetyMiddleware",
    "AzureContentSafetyImageMiddleware",
    "AzureProtectedMaterialMiddleware",
    "AzurePromptShieldMiddleware",
    "ContentSafetyViolationError",
    "_AzureContentSafetyBaseMiddleware",
    "_ContentSafetyState",
]
