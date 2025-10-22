"""Test deprecation and experimental utilities."""

import warnings
from typing import Any

import pytest

from langchain_azure_ai._api.base import (
    ExperimentalWarning,
    deprecated,
    experimental,
    get_deprecation_message,
    get_experimental_message,
    is_deprecated,
    is_experimental,
    warn_deprecated,
    warn_experimental,
)


def test_deprecated_function():
    """Test deprecation decorator on functions."""

    @deprecated("0.1.0", alternative="new_function")
    def old_function():
        return "old"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        result = old_function()

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "old_function is deprecated" in str(w[0].message)
        assert "new_function" in str(w[0].message)
        assert result == "old"


def test_experimental_function():
    """Test experimental decorator on functions."""

    @experimental()
    def experimental_function():
        return "experimental"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ExperimentalWarning)
        result = experimental_function()

        assert len(w) == 1
        assert issubclass(w[0].category, ExperimentalWarning)
        assert (
            "experimental_function is currently in preview and is subject to change"
            in str(w[0].message)
        )
        assert result == "experimental"


def test_experimental_class():
    """Test experimental decorator on classes."""

    @experimental(addendum="Requires experimental features enabled")
    class ExperimentalClass:
        def __init__(self):
            self.value = "experimental"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ExperimentalWarning)
        instance = ExperimentalClass()

        assert len(w) == 1
        assert issubclass(w[0].category, ExperimentalWarning)
        assert "experimental features enabled" in str(w[0].message)
        assert instance.value == "experimental"


def test_experimental_silent():
    """Test experimental decorator with warnings disabled."""

    @experimental(warn_on_use=False)
    def silent_experimental_function():
        return "silent"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ExperimentalWarning)
        result = silent_experimental_function()

        # Should be no warnings
        experimental_warnings = [
            warning
            for warning in w
            if issubclass(warning.category, ExperimentalWarning)
        ]
        assert len(experimental_warnings) == 0
        assert result == "silent"


def test_warn_experimental():
    """Test manual experimental warning."""

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ExperimentalWarning)
        warn_experimental(
            "experimental_feature", addendum="Enable with --experimental flag"
        )

        assert len(w) == 1
        assert issubclass(w[0].category, ExperimentalWarning)
        assert "--experimental flag" in str(w[0].message)


def test_is_experimental_check():
    """Test checking if objects are marked as experimental."""

    @experimental()
    class ExperimentalClass:
        pass

    @experimental(warn_on_use=False)
    def experimental_function():
        pass

    class RegularClass:
        pass

    def regular_function():
        pass

    assert is_experimental(ExperimentalClass)
    assert is_experimental(experimental_function)
    assert not is_experimental(RegularClass)
    assert not is_experimental(regular_function)


def test_is_deprecated_check():
    """Test checking if objects are marked as deprecated."""

    @deprecated("0.1.0")
    class DeprecatedClass:
        pass

    @deprecated("0.1.0")
    def deprecated_function():
        pass

    class RegularClass:
        pass

    def regular_function():
        pass

    assert is_deprecated(DeprecatedClass)
    assert is_deprecated(deprecated_function)
    assert not is_deprecated(RegularClass)
    assert not is_deprecated(regular_function)


def test_get_messages():
    """Test getting experimental and deprecation messages."""

    @experimental(message="Custom experimental message")
    def experimental_func():
        pass

    @deprecated("0.1.0", message="Custom deprecation message")
    def deprecated_func():
        pass

    def regular_func():
        pass

    assert get_experimental_message(experimental_func) == "Custom experimental message"
    assert get_deprecation_message(deprecated_func) == "Custom deprecation message"
    assert get_experimental_message(regular_func) is None
    assert get_deprecation_message(regular_func) is None


def test_custom_experimental_message():
    """Test custom experimental messages."""

    @experimental(message="This is a completely custom experimental message.")
    def custom_experimental_function():
        return "custom"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ExperimentalWarning)
        custom_experimental_function()

        assert len(w) == 1
        assert str(w[0].message) == "This is a completely custom experimental message."


def test_deprecated_class():
    """Test deprecation decorator on classes."""

    @deprecated("0.1.0", alternative="NewClass", removal="1.0.0")
    class OldClass:
        def __init__(self):
            self.value = "old"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        instance = OldClass()

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "OldClass is deprecated" in str(w[0].message)
        assert "NewClass" in str(w[0].message)
        assert "1.0.0" in str(w[0].message)
        assert instance.value == "old"


def test_warn_deprecated():
    """Test manual deprecation warning."""

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        warn_deprecated(
            "some_object",
            "0.2.0",
            alternative="new_object",
            addendum="Additional context here.",
        )

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "some_object is deprecated" in str(w[0].message)
        assert "new_object" in str(w[0].message)
        assert "Additional context" in str(w[0].message)


def test_pending_deprecation():
    """Test pending deprecation warnings."""

    @deprecated("0.3.0", pending=True, alternative="future_function")
    def soon_deprecated_function():
        return "soon"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore", DeprecationWarning)
        result = soon_deprecated_function()

        assert len(w) == 1
        assert issubclass(w[0].category, PendingDeprecationWarning)
        assert "will be deprecated" in str(w[0].message)
        assert result == "soon"
