"""Deprecation and experimental utilities for LangChain Azure AI.

This module provides decorators and utilities for marking classes, methods, and
functions as deprecated or in experimental status. It includes:

- @deprecated() decorator for marking features as deprecated
- @experimental() decorator for marking features as experimental
- warn_deprecated() and warn_experimental() functions for manual warnings
- Helper functions to check and retrieve deprecation/experimental status
- Warning management functions to suppress or surface warnings
"""

import functools
import inspect
import logging
import warnings
from typing import Any, Callable, Optional, Type, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Callable[..., Any])


class ExperimentalWarning(UserWarning):
    """Warning category for experimental features.

    Used to distinguish experimental warnings from other warning types.
    """

    pass


def deprecated(
    since: str,
    *,
    message: Optional[str] = None,
    name: Optional[str] = None,
    alternative: Optional[str] = None,
    pending: bool = False,
    removal: Optional[str] = None,
    addendum: Optional[str] = None,
) -> Callable[[T], T]:
    """Decorator to mark functions, methods, and classes as deprecated.

    Args:
        since: The LangChain Azure AI version when the deprecation started.
        message: Custom deprecation message. If not provided, a default message
            will be generated.
        name: The name of the deprecated object. If not provided, it will be
            inferred from the decorated object.
        alternative: The alternative to use instead of the deprecated object.
        pending: Whether this is a pending deprecation (default: False).
        removal: The version when the deprecated object will be removed.
        addendum: Additional information to add to the deprecation message.

    Returns:
        The decorated function, method, or class with deprecation warnings.

    Example:
        ```python
        @deprecated("0.2.0", alternative="NewClass", removal="1.0.0")
        class OldClass:
            pass

        @deprecated("0.1.5", message="Use new_function() instead")
        def old_function():
            pass
        ```
    """

    def decorator(obj: T) -> T:
        # Get the name of the deprecated object
        deprecated_name = name or _get_object_name(obj)

        # Generate the deprecation message
        warning_message = _create_deprecation_message(
            deprecated_name,
            since,
            message,
            alternative,
            pending,
            removal,
            addendum,
        )

        if inspect.isclass(obj):
            return _deprecate_class(obj, warning_message, pending)  # type: ignore[return-value]
        elif inspect.isfunction(obj) or inspect.ismethod(obj):
            return _deprecate_function(obj, warning_message, pending)  # type: ignore[return-value]
        else:
            # For other objects, just add a deprecation warning when accessed
            warnings.warn(
                warning_message,
                DeprecationWarning if not pending else PendingDeprecationWarning,
                stacklevel=2,
            )
            return obj

    return decorator


def experimental(
    *,
    message: Optional[str] = None,
    name: Optional[str] = None,
    warn_on_use: bool = True,
    addendum: Optional[str] = None,
) -> Callable[[T], T]:
    """Decorator to mark functions, methods, and classes as experimental.

    Experimental features are functional but may have breaking changes in future
    versions without following normal deprecation cycles.

    Args:
        message: Custom experimental message. If not provided, a default message
            will be generated.
        name: The name of the experimental object. If not provided, it will be
            inferred from the decorated object.
        warn_on_use: Whether to show a warning when the experimental feature is used.
            Defaults to True.
        addendum: Additional information to add to the experimental message.

    Returns:
        The decorated function, method, or class with experimental warnings.

    Example:
        ```python
        @experimental()
        class ExperimentalClass:
            pass

        @experimental(message="This feature is experimental and may change")
        def experimental_function():
            pass

        @experimental(warn_on_use=False, addendum="Enable with --experimental-features")
        def quiet_experimental_function():
            pass
        ```
    """

    def decorator(obj: T) -> T:
        # Get the name of the experimental object
        experimental_name = name or _get_object_name(obj)

        # Generate the experimental message
        warning_message = _create_experimental_message(
            experimental_name,
            message,
            addendum,
        )

        if inspect.isclass(obj):
            return _experimental_class(obj, warning_message, warn_on_use)  # type: ignore[return-value]
        elif inspect.isfunction(obj) or inspect.ismethod(obj):
            return _experimental_function(obj, warning_message, warn_on_use)  # type: ignore[return-value]
        else:
            if warn_on_use:
                warnings.warn(
                    warning_message,
                    ExperimentalWarning,
                    stacklevel=2,
                )
            # Add experimental info to the object
            if hasattr(obj, "__dict__"):
                obj_cast: Any = cast(Any, obj)
                obj_cast.__experimental__ = True
                obj_cast.__experimental_message__ = warning_message
            return obj

    return decorator


def _get_object_name(obj: Any) -> str:
    """Get the name of an object.

    Args:
        obj: The object to get the name from.

    Returns:
        The name of the object, or its string representation.
    """
    if hasattr(obj, "__name__"):
        return obj.__name__
    elif hasattr(obj, "__class__"):
        return obj.__class__.__name__
    else:
        return str(obj)


def _create_deprecation_message(
    name: str,
    since: str,
    message: Optional[str],
    alternative: Optional[str],
    pending: bool,
    removal: Optional[str],
    addendum: Optional[str],
) -> str:
    """Create a standardized deprecation message.

    Args:
        name: The name of the deprecated object.
        since: The version when deprecation started.
        message: Custom message, if provided.
        alternative: The alternative to use instead.
        pending: Whether this is a pending deprecation.
        removal: The version when the object will be removed.
        addendum: Additional information to append.

    Returns:
        A formatted deprecation message.
    """
    if message:
        warning_message = message
    else:
        deprecation_type = "will be deprecated" if pending else "is deprecated"
        warning_message = f"{name} {deprecation_type} as of langchain-azure-ai=={since}"

        if removal:
            warning_message += f" and will be removed in {removal}"

        if alternative:
            warning_message += f". Use {alternative} instead"

        warning_message += "."

    if addendum:
        warning_message += f" {addendum}"

    return warning_message


def _create_experimental_message(
    name: str,
    message: Optional[str],
    addendum: Optional[str],
) -> str:
    """Create a standardized experimental message.

    Args:
        name: The name of the experimental object.
        message: Custom message, if provided.
        addendum: Additional information to append.

    Returns:
        A formatted experimental warning message.
    """
    if message:
        warning_message = message
    else:
        warning_message = (
            f"{name} is currently in preview and is subject to change. This preview "
            "is provided without a service-level agreement, and we don't recommend "
            "it for production workloads. Certain features might not be supported or "
            "might have constrained capabilities. For more information, see "
            "https://azure.microsoft.com/support/legal/preview-supplemental-terms"
        )

    if addendum:
        warning_message += f" {addendum}"

    return warning_message


def _deprecate_class(cls: Type[Any], warning_message: str, pending: bool) -> Type[Any]:
    """Add deprecation warning to a class.

    Args:
        cls: The class to deprecate.
        warning_message: The deprecation warning message.
        pending: Whether this is a pending deprecation.

    Returns:
        The decorated class with deprecation warnings.
    """
    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            warning_message,
            DeprecationWarning if not pending else PendingDeprecationWarning,
            stacklevel=2,
        )
        original_init(self, *args, **kwargs)

    cls.__init__ = __init__  # type: ignore[method-assign]

    # Add deprecation info to the class
    cls.__deprecated__ = True  # type: ignore[attr-defined]
    cls.__deprecation_message__ = warning_message  # type: ignore[attr-defined]

    return cls


def _deprecate_function(
    func: Callable[..., Any], warning_message: str, pending: bool
) -> Callable[..., Any]:
    """Add deprecation warning to a function.

    Args:
        func: The function to deprecate.
        warning_message: The deprecation warning message.
        pending: Whether this is a pending deprecation.

    Returns:
        The decorated function with deprecation warnings.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            warning_message,
            DeprecationWarning if not pending else PendingDeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    # Add deprecation info to the function
    wrapper.__deprecated__ = True  # type: ignore[attr-defined]
    wrapper.__deprecation_message__ = warning_message  # type: ignore[attr-defined]

    return cast(Callable[..., Any], wrapper)


def _experimental_class(
    cls: Type[Any], warning_message: str, warn_on_use: bool
) -> Type[Any]:
    """Add experimental warning to a class.

    Args:
        cls: The class to mark as experimental.
        warning_message: The experimental warning message.
        warn_on_use: Whether to warn when the class is instantiated.

    Returns:
        The decorated class with experimental warnings.
    """
    if warn_on_use:
        original_init = cls.__init__

        @functools.wraps(original_init)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            warnings.warn(
                warning_message,
                ExperimentalWarning,
                stacklevel=2,
            )
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__  # type: ignore[method-assign]

    # Add experimental info to the class
    cls.__experimental__ = True  # type: ignore[attr-defined]
    cls.__experimental_message__ = warning_message  # type: ignore[attr-defined]

    return cls


def _experimental_function(
    func: Callable[..., Any], warning_message: str, warn_on_use: bool
) -> Callable[..., Any]:
    """Add experimental warning to a function.

    Args:
        func: The function to mark as experimental.
        warning_message: The experimental warning message.
        warn_on_use: Whether to warn when the function is called.

    Returns:
        The decorated function with experimental warnings.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if warn_on_use:
            warnings.warn(
                warning_message,
                ExperimentalWarning,
                stacklevel=2,
            )
        return func(*args, **kwargs)

    # Add experimental info to the function
    wrapper.__experimental__ = True  # type: ignore[attr-defined]
    wrapper.__experimental_message__ = warning_message  # type: ignore[attr-defined]

    return cast(Callable[..., Any], wrapper)


def warn_deprecated(
    object_name: str,
    since: str,
    *,
    message: Optional[str] = None,
    alternative: Optional[str] = None,
    pending: bool = False,
    removal: Optional[str] = None,
    addendum: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """Issue a deprecation warning for an object.

    This is useful for deprecating objects that can't use the decorator,
    such as module-level variables or dynamic objects.

    Args:
        object_name: The name of the deprecated object.
        since: The LangChain Azure AI version when the deprecation started.
        message: Custom deprecation message.
        alternative: The alternative to use instead.
        pending: Whether this is a pending deprecation.
        removal: The version when the object will be removed.
        addendum: Additional information.
        stacklevel: The stack level for the warning.
    """
    warning_message = _create_deprecation_message(
        object_name, since, message, alternative, pending, removal, addendum
    )

    warnings.warn(
        warning_message,
        DeprecationWarning if not pending else PendingDeprecationWarning,
        stacklevel=stacklevel,
    )


def warn_experimental(
    object_name: str,
    *,
    message: Optional[str] = None,
    addendum: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """Issue an experimental warning for an object.

    This is useful for warning about experimental objects that can't use the decorator,
    such as module-level variables or dynamic objects.

    Args:
        object_name: The name of the experimental object.
        message: Custom experimental message.
        addendum: Additional information.
        stacklevel: The stack level for the warning.
    """
    warning_message = _create_experimental_message(object_name, message, addendum)

    warnings.warn(
        warning_message,
        ExperimentalWarning,
        stacklevel=stacklevel,
    )


def surface_deprecation_warnings() -> None:
    """Ensure that deprecation warnings are shown to users.

    LangChain Azure AI deprecation warnings are shown by default.
    This function is provided for completeness and to allow users to
    explicitly enable deprecation warnings if they have been disabled.
    """
    warnings.filterwarnings(
        "default", category=DeprecationWarning, module="langchain_azure_ai"
    )
    warnings.filterwarnings(
        "default", category=PendingDeprecationWarning, module="langchain_azure_ai"
    )


def suppress_deprecation_warnings() -> None:
    """Suppress LangChain Azure AI deprecation warnings.

    This can be useful during testing or when using deprecated functionality
    that you're not ready to migrate yet.
    """
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="langchain_azure_ai"
    )
    warnings.filterwarnings(
        "ignore", category=PendingDeprecationWarning, module="langchain_azure_ai"
    )


def surface_experimental_warnings() -> None:
    """Ensure that experimental warnings are shown to users.

    LangChain Azure AI experimental warnings are shown by default.
    This function is provided for completeness and to allow users to
    explicitly enable experimental warnings if they have been disabled.
    """
    warnings.filterwarnings(
        "default", category=ExperimentalWarning, module="langchain_azure_ai"
    )


def suppress_experimental_warnings() -> None:
    """Suppress LangChain Azure AI experimental warnings.

    This can be useful during testing or when using experimental functionality
    and you don't want to see the warnings.
    """
    warnings.filterwarnings(
        "ignore", category=ExperimentalWarning, module="langchain_azure_ai"
    )


def is_experimental(obj: Any) -> bool:
    """Check if an object is marked as experimental.

    Args:
        obj: The object to check.

    Returns:
        True if the object is marked as experimental, False otherwise.
    """
    return getattr(obj, "__experimental__", False)


def is_deprecated(obj: Any) -> bool:
    """Check if an object is marked as deprecated.

    Args:
        obj: The object to check.

    Returns:
        True if the object is marked as deprecated, False otherwise.
    """
    return getattr(obj, "__deprecated__", False)


def get_experimental_message(obj: Any) -> Optional[str]:
    """Get the experimental message for an object.

    Args:
        obj: The object to get the experimental message for.

    Returns:
        The experimental message if the object is marked as experimental,
        None otherwise.
    """
    return getattr(obj, "__experimental_message__", None)


def get_deprecation_message(obj: Any) -> Optional[str]:
    """Get the deprecation message for an object.

    Args:
        obj: The object to get the deprecation message for.

    Returns:
        The deprecation message if the object is marked as deprecated,
        None otherwise.
    """
    return getattr(obj, "__deprecation_message__", None)
