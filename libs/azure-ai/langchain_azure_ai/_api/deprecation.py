"""Deprecation and beta utilities for LangChain Azure AI."""

import functools
import inspect
import logging
import warnings
from typing import Any, Callable, Optional, Type, TypeVar, Union, cast

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Callable[..., Any])


class BetaWarning(UserWarning):
    """Warning category for beta features."""
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
            return _deprecate_class(obj, warning_message, pending)
        elif inspect.isfunction(obj) or inspect.ismethod(obj):
            return _deprecate_function(obj, warning_message, pending)
        else:
            # For other objects, just add a deprecation warning when accessed
            warnings.warn(
                warning_message,
                DeprecationWarning if not pending else PendingDeprecationWarning,
                stacklevel=2,
            )
            return obj

    return decorator


def beta(
    *,
    message: Optional[str] = None,
    name: Optional[str] = None,
    warn_on_use: bool = True,
    addendum: Optional[str] = None,
) -> Callable[[T], T]:
    """Decorator to mark functions, methods, and classes as beta/experimental.

    Beta features are functional but may have breaking changes in future versions
    without following normal deprecation cycles.

    Args:
        message: Custom beta message. If not provided, a default message
               will be generated.
        name: The name of the beta object. If not provided, it will be
              inferred from the decorated object.
        warn_on_use: Whether to show a warning when the beta feature is used.
                    Defaults to True.
        addendum: Additional information to add to the beta message.

    Returns:
        The decorated function, method, or class with beta warnings.

    Example:
        ```python
        @beta()
        class ExperimentalClass:
            pass

        @beta(message="This feature is experimental and may change")
        def experimental_function():
            pass

        @beta(warn_on_use=False, addendum="Enable with --experimental-features")
        def quiet_beta_function():
            pass
        ```
    """

    def decorator(obj: T) -> T:
        # Get the name of the beta object
        beta_name = name or _get_object_name(obj)

        # Generate the beta message
        warning_message = _create_beta_message(
            beta_name,
            message,
            addendum,
        )

        if inspect.isclass(obj):
            return _beta_class(obj, warning_message, warn_on_use)
        elif inspect.isfunction(obj) or inspect.ismethod(obj):
            return _beta_function(obj, warning_message, warn_on_use)
        else:
            # For other objects, just add a beta warning when accessed if enabled
            if warn_on_use:
                warnings.warn(
                    warning_message,
                    BetaWarning,
                    stacklevel=2,
                )
            # Add beta info to the object
            if hasattr(obj, '__dict__'):
                obj.__beta__ = True
                obj.__beta_message__ = warning_message
            return obj

    return decorator


def _get_object_name(obj: Any) -> str:
    """Get the name of an object."""
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
    """Create a standardized deprecation message."""
    if message:
        warning_message = message
    else:
        deprecation_type = "will be deprecated" if pending else "is deprecated"
        warning_message = f"{name} {deprecation_type} as of langchain-azure-ai=={since}"

        if alternative:
            warning_message += f". Use {alternative} instead"

        if removal:
            warning_message += f" and will be removed in {removal}"

        warning_message += "."

    if addendum:
        warning_message += f" {addendum}"

    return warning_message


def _create_beta_message(
    name: str,
    message: Optional[str],
    addendum: Optional[str],
) -> str:
    """Create a standardized beta message."""
    if message:
        warning_message = message
    else:
        warning_message = (
            f"{name} is in beta. It is actively being worked on, so the API may change "
            "in future versions without following normal deprecation cycles."
        )

    if addendum:
        warning_message += f" {addendum}"

    return warning_message


def _deprecate_class(
    cls: Type[Any], warning_message: str, pending: bool
) -> Type[Any]:
    """Add deprecation warning to a class."""
    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            warning_message,
            DeprecationWarning if not pending else PendingDeprecationWarning,
            stacklevel=2,
        )
        original_init(self, *args, **kwargs)

    cls.__init__ = __init__

    # Add deprecation info to the class
    cls.__deprecated__ = True
    cls.__deprecation_message__ = warning_message

    return cls


def _deprecate_function(func: Callable[..., Any], warning_message: str, pending: bool) -> Callable[..., Any]:
    """Add deprecation warning to a function."""

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


def _beta_class(
    cls: Type[Any], warning_message: str, warn_on_use: bool
) -> Type[Any]:
    """Add beta warning to a class."""
    if warn_on_use:
        original_init = cls.__init__

        @functools.wraps(original_init)
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            warnings.warn(
                warning_message,
                BetaWarning,
                stacklevel=2,
            )
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__

    # Add beta info to the class
    cls.__beta__ = True
    cls.__beta_message__ = warning_message

    return cls


def _beta_function(func: Callable[..., Any], warning_message: str, warn_on_use: bool) -> Callable[..., Any]:
    """Add beta warning to a function."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if warn_on_use:
            warnings.warn(
                warning_message,
                BetaWarning,
                stacklevel=2,
            )
        return func(*args, **kwargs)

    # Add beta info to the function
    wrapper.__beta__ = True  # type: ignore[attr-defined]
    wrapper.__beta_message__ = warning_message  # type: ignore[attr-defined]

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


def warn_beta(
    object_name: str,
    *,
    message: Optional[str] = None,
    addendum: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """Issue a beta warning for an object.

    This is useful for warning about beta objects that can't use the decorator,
    such as module-level variables or dynamic objects.

    Args:
        object_name: The name of the beta object.
        message: Custom beta message.
        addendum: Additional information.
        stacklevel: The stack level for the warning.
    """
    warning_message = _create_beta_message(object_name, message, addendum)

    warnings.warn(
        warning_message,
        BetaWarning,
        stacklevel=stacklevel,
    )


def surface_langchain_azure_ai_deprecation_warnings() -> None:
    """Ensure that deprecation warnings are shown to users.

    LangChain Azure AI deprecation warnings are shown by default.
    This function is provided for completeness and to allow users to
    explicitly enable deprecation warnings if they have been disabled.
    """
    warnings.filterwarnings("default", category=DeprecationWarning, module="langchain_azure_ai")
    warnings.filterwarnings("default", category=PendingDeprecationWarning, module="langchain_azure_ai")


def suppress_langchain_azure_ai_deprecation_warnings() -> None:
    """Suppress LangChain Azure AI deprecation warnings.

    This can be useful during testing or when using deprecated functionality
    that you're not ready to migrate yet.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_azure_ai")
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="langchain_azure_ai")


def surface_langchain_azure_ai_beta_warnings() -> None:
    """Ensure that beta warnings are shown to users.

    LangChain Azure AI beta warnings are shown by default.
    This function is provided for completeness and to allow users to
    explicitly enable beta warnings if they have been disabled.
    """
    warnings.filterwarnings("default", category=BetaWarning, module="langchain_azure_ai")


def suppress_langchain_azure_ai_beta_warnings() -> None:
    """Suppress LangChain Azure AI beta warnings.

    This can be useful during testing or when using beta functionality
    and you don't want to see the warnings.
    """
    warnings.filterwarnings("ignore", category=BetaWarning, module="langchain_azure_ai")


def is_beta(obj: Any) -> bool:
    """Check if an object is marked as beta.

    Args:
        obj: The object to check.

    Returns:
        True if the object is marked as beta, False otherwise.
    """
    return getattr(obj, "__beta__", False)


def is_deprecated(obj: Any) -> bool:
    """Check if an object is marked as deprecated.

    Args:
        obj: The object to check.

    Returns:
        True if the object is marked as deprecated, False otherwise.
    """
    return getattr(obj, "__deprecated__", False)


def get_beta_message(obj: Any) -> Optional[str]:
    """Get the beta message for an object.

    Args:
        obj: The object to get the beta message for.

    Returns:
        The beta message if the object is marked as beta, None otherwise.
    """
    return getattr(obj, "__beta_message__", None)


def get_deprecation_message(obj: Any) -> Optional[str]:
    """Get the deprecation message for an object.

    Args:
        obj: The object to get the deprecation message for.

    Returns:
        The deprecation message if the object is marked as deprecated, None otherwise.
    """
    return getattr(obj, "__deprecation_message__", None)