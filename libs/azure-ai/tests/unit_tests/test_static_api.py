"""Tests for static API methods."""

import logging
import pytest

from langchain_azure_ai.tracing import AzureAIOpenTelemetryTracer


def test_set_app_insights():
    """Test setting Application Insights connection string."""
    conn_str = "InstrumentationKey=test-key;IngestionEndpoint=https://test.com"
    AzureAIOpenTelemetryTracer.set_app_insights(conn_str)
    
    assert AzureAIOpenTelemetryTracer._app_insights_connection_string == conn_str


def test_set_config():
    """Test setting configuration."""
    config = {
        "provider_name": "azure.ai.openai",
        "redact_messages": True,
        "sampling_rate": 0.5,
    }
    
    AzureAIOpenTelemetryTracer.set_config(config)
    
    current_config = AzureAIOpenTelemetryTracer.get_config()
    assert current_config["provider_name"] == "azure.ai.openai"
    assert current_config["redact_messages"] is True
    assert current_config["sampling_rate"] == 0.5


def test_set_config_validation():
    """Test that invalid configuration raises error."""
    with pytest.raises(ValueError):
        AzureAIOpenTelemetryTracer.set_config({"sampling_rate": 2.0})


def test_get_config():
    """Test getting configuration."""
    AzureAIOpenTelemetryTracer.set_config({"provider_name": "test"})
    config = AzureAIOpenTelemetryTracer.get_config()
    
    assert isinstance(config, dict)
    assert "provider_name" in config
    assert config["provider_name"] == "test"


def test_is_active_initially_false():
    """Test that is_active is False initially."""
    # Reset state
    AzureAIOpenTelemetryTracer.shutdown()
    
    assert AzureAIOpenTelemetryTracer.is_active() is False


def test_autolog_activates(monkeypatch):
    """Test that autolog activates the tracer."""
    # Reset state
    AzureAIOpenTelemetryTracer.shutdown()
    
    # Mock monkey patch application to avoid actual patching
    def mock_apply_patches():
        pass
    
    monkeypatch.setattr(
        "langchain_azure_ai.tracing.tracer.apply_monkey_patches",
        mock_apply_patches
    )
    
    # Set connection string
    AzureAIOpenTelemetryTracer.set_app_insights("InstrumentationKey=test")
    
    # Enable autolog with monkey patching
    AzureAIOpenTelemetryTracer.set_config({"patch_mode": "monkey"})
    AzureAIOpenTelemetryTracer.autolog()
    
    assert AzureAIOpenTelemetryTracer.is_active() is True
    assert AzureAIOpenTelemetryTracer.get_tracer_instance() is not None


def test_autolog_idempotent(monkeypatch, caplog):
    """Test that calling autolog multiple times is safe."""
    # Reset state
    AzureAIOpenTelemetryTracer.shutdown()
    
    # Mock monkey patch application
    def mock_apply_patches():
        pass
    
    monkeypatch.setattr(
        "langchain_azure_ai.tracing.tracer.apply_monkey_patches",
        mock_apply_patches
    )
    
    AzureAIOpenTelemetryTracer.set_app_insights("InstrumentationKey=test")
    AzureAIOpenTelemetryTracer.set_config({"patch_mode": "monkey"})
    
    with caplog.at_level(logging.INFO):
        AzureAIOpenTelemetryTracer.autolog()
        AzureAIOpenTelemetryTracer.autolog()  # Second call
    
    # Should log that it's already active
    assert any("already active" in record.message.lower() for record in caplog.records)


def test_autolog_with_overrides(monkeypatch):
    """Test autolog with configuration overrides."""
    # Reset state
    AzureAIOpenTelemetryTracer.shutdown()
    
    # Mock monkey patch application
    def mock_apply_patches():
        pass
    
    monkeypatch.setattr(
        "langchain_azure_ai.tracing.tracer.apply_monkey_patches",
        mock_apply_patches
    )
    
    AzureAIOpenTelemetryTracer.set_app_insights("InstrumentationKey=test")
    AzureAIOpenTelemetryTracer.set_config({"provider_name": "original"})
    
    # Override with autolog
    AzureAIOpenTelemetryTracer.autolog(
        provider_name="overridden",
        sampling_rate=0.5,
        patch_mode="monkey"
    )
    
    config = AzureAIOpenTelemetryTracer.get_config()
    assert config["provider_name"] == "overridden"
    assert config["sampling_rate"] == 0.5


def test_add_tags():
    """Test adding global tags."""
    AzureAIOpenTelemetryTracer.add_tags({
        "environment": "test",
        "version": "1.0.0"
    })
    
    # Note: Accessing private attributes for testing purposes
    # In production code, use public methods to query state
    assert AzureAIOpenTelemetryTracer._global_tags["environment"] == "test"
    assert AzureAIOpenTelemetryTracer._global_tags["version"] == "1.0.0"


def test_update_redaction_rules():
    """Test updating redaction rules."""
    AzureAIOpenTelemetryTracer.update_redaction_rules(
        redact_messages=True,
        redact_tool_arguments=True
    )
    
    config = AzureAIOpenTelemetryTracer.get_config()
    assert config["redact_messages"] is True
    assert config["redact_tool_arguments"] is True


def test_force_flush():
    """Test force flush doesn't raise errors."""
    # Should not raise even if nothing is active
    AzureAIOpenTelemetryTracer.force_flush()


def test_shutdown(monkeypatch):
    """Test shutdown cleans up properly."""
    # Mock monkey patch functions
    def mock_apply_patches():
        pass
    
    def mock_is_patched():
        return True
    
    def mock_remove_patches():
        pass
    
    monkeypatch.setattr(
        "langchain_azure_ai.tracing.tracer.apply_monkey_patches",
        mock_apply_patches
    )
    monkeypatch.setattr(
        "langchain_azure_ai.tracing.tracer.is_monkey_patched",
        mock_is_patched
    )
    monkeypatch.setattr(
        "langchain_azure_ai.tracing.tracer.remove_monkey_patches",
        mock_remove_patches
    )
    
    # Activate
    AzureAIOpenTelemetryTracer.set_app_insights("InstrumentationKey=test")
    AzureAIOpenTelemetryTracer.set_config({"patch_mode": "monkey"})
    AzureAIOpenTelemetryTracer.autolog()
    
    assert AzureAIOpenTelemetryTracer.is_active() is True
    
    # Shutdown
    AzureAIOpenTelemetryTracer.shutdown()
    
    assert AzureAIOpenTelemetryTracer.is_active() is False
    assert AzureAIOpenTelemetryTracer.get_tracer_instance() is None


def test_get_tracer_instance():
    """Test getting tracer instance."""
    # Reset state
    AzureAIOpenTelemetryTracer.shutdown()
    
    # Initially None
    assert AzureAIOpenTelemetryTracer.get_tracer_instance() is None
    
    # After autolog, should have instance
    # (We'll skip actual activation to avoid dependencies)


def test_autolog_without_connection_string(monkeypatch, caplog):
    """Test autolog warns if no connection string is set."""
    # Reset state
    AzureAIOpenTelemetryTracer.shutdown()
    AzureAIOpenTelemetryTracer._app_insights_connection_string = None
    
    # Clear env var
    monkeypatch.delenv("APPLICATION_INSIGHTS_CONNECTION_STRING", raising=False)
    
    # Mock monkey patch application
    def mock_apply_patches():
        pass
    
    monkeypatch.setattr(
        "langchain_azure_ai.tracing.tracer.apply_monkey_patches",
        mock_apply_patches
    )
    
    AzureAIOpenTelemetryTracer.set_config({"patch_mode": "monkey"})
    
    with caplog.at_level(logging.WARNING):
        AzureAIOpenTelemetryTracer.autolog()
    
    # Should warn about missing connection string
    assert any(
        "No Application Insights connection string" in record.message
        for record in caplog.records
    )


def test_autolog_with_env_var_connection_string(monkeypatch):
    """Test autolog uses environment variable for connection string."""
    # Reset state
    AzureAIOpenTelemetryTracer.shutdown()
    AzureAIOpenTelemetryTracer._app_insights_connection_string = None
    
    # Set env var
    conn_str = "InstrumentationKey=env-test"
    monkeypatch.setenv("APPLICATION_INSIGHTS_CONNECTION_STRING", conn_str)
    
    # Mock monkey patch application
    def mock_apply_patches():
        pass
    
    monkeypatch.setattr(
        "langchain_azure_ai.tracing.tracer.apply_monkey_patches",
        mock_apply_patches
    )
    
    AzureAIOpenTelemetryTracer.set_config({"patch_mode": "monkey"})
    AzureAIOpenTelemetryTracer.autolog()
    
    assert AzureAIOpenTelemetryTracer._app_insights_connection_string == conn_str
