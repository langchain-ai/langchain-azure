"""Test deprecation and beta utilities."""

import warnings
from typing import Any

import pytest

from langchain_azure_ai._api.deprecation import (
    BetaWarning,
    beta,
    deprecated,
    get_beta_message,
    get_deprecation_message,
    is_beta,
    is_deprecated,
    suppress_langchain_azure_ai_beta_warnings,
    suppress_langchain_azure_ai_deprecation_warnings,
    surface_langchain_azure_ai_beta_warnings,
    surface_langchain_azure_ai_deprecation_warnings,
    warn_beta,
    warn_deprecated,
)


def test_deprecated_function():
    """Test deprecation decorator on functions."""

    @deprecated("0.1.0", alternative="new_function")
    def old_function():
        return "old"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = old_function()

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "old_function is deprecated" in str(w[0].message)
        assert "new_function" in str(w[0].message)
        assert result == "old"


def test_beta_function():
    """Test beta decorator on functions."""

    @beta()
    def experimental_function():
        return "experimental"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = experimental_function()

        assert len(w) == 1
        assert issubclass(w[0].category, BetaWarning)
        assert "experimental_function is in beta" in str(w[0].message)
        assert "API may change" in str(w[0].message)
        assert result == "experimental"


def test_beta_class():
    """Test beta decorator on classes."""

    @beta(addendum="Requires experimental features enabled")
    class ExperimentalClass:
        def __init__(self):
            self.value = "experimental"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        instance = ExperimentalClass()

        assert len(w) == 1
        assert issubclass(w[0].category, BetaWarning)
        assert "ExperimentalClass is in beta" in str(w[0].message)
        assert "experimental features enabled" in str(w[0].message)
        assert instance.value == "experimental"


def test_beta_silent():
    """Test beta decorator with warnings disabled."""

    @beta(warn_on_use=False)
    def silent_beta_function():
        return "silent"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = silent_beta_function()

        # Should be no warnings
        beta_warnings = [
            warning for warning in w if issubclass(warning.category, BetaWarning)
        ]
        assert len(beta_warnings) == 0
        assert result == "silent"


def test_warn_beta():
    """Test manual beta warning."""

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_beta("experimental_feature", addendum="Enable with --experimental flag")

        assert len(w) == 1
        assert issubclass(w[0].category, BetaWarning)
        assert "experimental_feature is in beta" in str(w[0].message)
        assert "--experimental flag" in str(w[0].message)


def test_is_beta_check():
    """Test checking if objects are marked as beta."""

    @beta()
    class BetaClass:
        pass

    @beta(warn_on_use=False)
    def beta_function():
        pass

    class RegularClass:
        pass

    def regular_function():
        pass

    assert is_beta(BetaClass)
    assert is_beta(beta_function)
    assert not is_beta(RegularClass)
    assert not is_beta(regular_function)


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
    """Test getting beta and deprecation messages."""

    @beta(message="Custom beta message")
    def beta_func():
        pass

    @deprecated("0.1.0", message="Custom deprecation message")
    def deprecated_func():
        pass

    def regular_func():
        pass

    assert get_beta_message(beta_func) == "Custom beta message"
    assert get_deprecation_message(deprecated_func) == "Custom deprecation message"
    assert get_beta_message(regular_func) is None
    assert get_deprecation_message(regular_func) is None


def test_suppress_beta_warnings():
    """Test suppressing beta warnings."""

    @beta()
    def beta_function():
        return "beta"

    suppress_langchain_azure_ai_beta_warnings()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        beta_function()

        # Should be no warnings due to filter
        beta_warnings = [
            warning for warning in w if issubclass(warning.category, BetaWarning)
        ]
        assert len(beta_warnings) == 0

    # Re-enable warnings
    surface_langchain_azure_ai_beta_warnings()


def test_custom_beta_message():
    """Test custom beta messages."""

    @beta(message="This is a completely custom beta message.")
    def custom_beta_function():
        return "custom"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        custom_beta_function()

        assert len(w) == 1
        assert str(w[0].message) == "This is a completely custom beta message."


# Keep existing deprecation tests...
def test_deprecated_class():
    """Test deprecation decorator on classes."""

    @deprecated("0.1.0", alternative="NewClass", removal="1.0.0")
    class OldClass:
        def __init__(self):
            self.value = "old"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
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
        warnings.simplefilter("always")
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
        warnings.simplefilter("always")
        result = soon_deprecated_function()

        assert len(w) == 1
        assert issubclass(w[0].category, PendingDeprecationWarning)
        assert "will be deprecated" in str(w[0].message)
        assert result == "soon"


def test_suppress_deprecation_warnings():
    """Test suppressing deprecation warnings."""

    @deprecated("0.1.0", alternative="new_function")
    def old_function():
        return "old"

    suppress_langchain_azure_ai_deprecation_warnings()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old_function()

        # Should be no warnings due to filter
        deprecation_warnings = [
            warning for warning in w if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0

    # Re-enable warnings
    surface_langchain_azure_ai_deprecation_warnings()
