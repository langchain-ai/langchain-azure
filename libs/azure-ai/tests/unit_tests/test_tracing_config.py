"""Tests for configuration management."""

from langchain_azure_ai.tracing.config import (
    DEFAULT_CONFIG,
    TracingConfig,
    validate_config,
)


def test_default_config():
    """Test that default configuration has expected values."""
    assert DEFAULT_CONFIG["provider_name"] is None
    assert DEFAULT_CONFIG["redact_messages"] is False
    assert DEFAULT_CONFIG["sampling_rate"] == 1.0
    assert DEFAULT_CONFIG["patch_mode"] == "callback"
    assert DEFAULT_CONFIG["aggregate_usage"] is True


def test_tracing_config_init():
    """Test TracingConfig initialization."""
    config = TracingConfig()
    assert config.get("provider_name") is None
    assert config.get("sampling_rate") == 1.0


def test_tracing_config_update():
    """Test updating configuration."""
    config = TracingConfig()
    config.update({"provider_name": "test_provider", "sampling_rate": 0.5})
    
    assert config.get("provider_name") == "test_provider"
    assert config.get("sampling_rate") == 0.5


def test_tracing_config_get_all():
    """Test getting all configuration values."""
    config = TracingConfig()
    config.update({"provider_name": "test"})
    
    all_config = config.get_all()
    assert isinstance(all_config, dict)
    assert all_config["provider_name"] == "test"
    assert "sampling_rate" in all_config


def test_tracing_config_reset():
    """Test resetting configuration to defaults."""
    config = TracingConfig()
    config.update({"provider_name": "test"})
    assert config.get("provider_name") == "test"
    
    config.reset()
    assert config.get("provider_name") is None


def test_validate_config_valid():
    """Test validation of valid configuration."""
    config = {
        "sampling_rate": 0.75,
        "patch_mode": "monkey",
        "max_message_characters": 1000,
    }
    # Should not raise
    validate_config(config)


def test_validate_config_invalid_sampling_rate():
    """Test validation fails for invalid sampling rate."""
    with pytest.raises(ValueError, match="sampling_rate must be between"):
        validate_config({"sampling_rate": 1.5})
    
    with pytest.raises(ValueError, match="sampling_rate must be between"):
        validate_config({"sampling_rate": -0.1})


def test_validate_config_invalid_patch_mode():
    """Test validation fails for invalid patch mode."""
    with pytest.raises(ValueError, match="patch_mode must be one of"):
        validate_config({"patch_mode": "invalid"})


def test_validate_config_invalid_max_message_characters():
    """Test validation fails for invalid max_message_characters."""
    with pytest.raises(ValueError, match="max_message_characters"):
        validate_config({"max_message_characters": -100})
    
    with pytest.raises(ValueError, match="max_message_characters"):
        validate_config({"max_message_characters": "invalid"})


def test_env_var_override(monkeypatch):
    """Test configuration from environment variables."""
    monkeypatch.setenv("AZURE_GENAI_REDACT_MESSAGES", "true")
    monkeypatch.setenv("AZURE_GENAI_SAMPLING_RATE", "0.5")
    monkeypatch.setenv("AZURE_GENAI_PROVIDER_NAME", "env_provider")
    
    config = TracingConfig()
    
    assert config.get("redact_messages") is True
    assert config.get("sampling_rate") == 0.5
    assert config.get("provider_name") == "env_provider"


def test_env_var_boolean_parsing(monkeypatch):
    """Test boolean environment variable parsing."""
    test_cases = [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ]
    
    for env_value, expected in test_cases:
        monkeypatch.setenv("AZURE_GENAI_REDACT_MESSAGES", env_value)
        config = TracingConfig()
        assert config.get("redact_messages") == expected


def test_config_thread_safety():
    """Test that config updates are thread-safe."""
    import threading
    
    config = TracingConfig()
    errors = []
    
    def update_config(value):
        try:
            config.update({"provider_name": f"provider_{value}"})
        except Exception as e:
            errors.append(e)
    
    threads = [threading.Thread(target=update_config, args=(i,)) for i in range(10)]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(errors) == 0
    assert config.get("provider_name").startswith("provider_")
