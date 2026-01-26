"""Configuration module for Azure AI Foundry integration.

This module provides configuration management for:
- Azure AI Foundry project settings
- Azure OpenAI settings
- Observability settings (Application Insights, LangSmith)
- Authentication settings

Configuration is loaded from environment variables with sensible defaults.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration."""

    endpoint: str = ""
    api_key: str = ""
    api_version: str = "2024-12-01-preview"
    deployment_name: str = "gpt-4o-mini"
    embedding_deployment: str = "text-embedding-ada-002"

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        """Load configuration from environment variables."""
        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
            embedding_deployment=os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
            ),
        )

    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.endpoint:
            logger.warning("AZURE_OPENAI_ENDPOINT not set")
            return False
        return True


@dataclass
class AzureAIFoundryConfig:
    """Azure AI Foundry project configuration."""

    project_endpoint: str = ""
    hub_name: str = ""
    project_name: str = ""
    location: str = "eastus"
    subscription_id: str = ""
    resource_group: str = ""

    @classmethod
    def from_env(cls) -> "AzureAIFoundryConfig":
        """Load configuration from environment variables."""
        return cls(
            project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT", ""),
            hub_name=os.getenv("AZURE_AI_HUB_NAME", ""),
            project_name=os.getenv("AZURE_AI_PROJECT_NAME", ""),
            location=os.getenv("AZURE_LOCATION", "eastus"),
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID", ""),
            resource_group=os.getenv("AZURE_RESOURCE_GROUP", ""),
        )

    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.project_endpoint:
            logger.warning("AZURE_AI_PROJECT_ENDPOINT not set - Foundry mode disabled")
            return False
        return True

    @property
    def is_enabled(self) -> bool:
        """Check if Azure AI Foundry is enabled."""
        return bool(self.project_endpoint)


@dataclass
class ObservabilityConfig:
    """Observability configuration for tracing and monitoring."""

    # Application Insights
    app_insights_connection_string: str = ""
    enable_app_insights: bool = False

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = ""
    enable_langsmith: bool = False

    # General tracing
    enable_tracing: bool = True
    trace_content: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Load configuration from environment variables."""
        app_insights_cs = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
        langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")

        return cls(
            app_insights_connection_string=app_insights_cs,
            enable_app_insights=bool(app_insights_cs),
            langsmith_api_key=langsmith_key,
            langsmith_project=os.getenv("LANGCHAIN_PROJECT", ""),
            enable_langsmith=os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true",
            enable_tracing=os.getenv("ENABLE_TRACING", "true").lower() == "true",
            trace_content=os.getenv("TRACE_CONTENT", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def configure_logging(self) -> None:
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def setup_tracing(self) -> None:
        """Set up tracing based on configuration."""
        if self.enable_app_insights and self.app_insights_connection_string:
            self._setup_app_insights()

        if self.enable_langsmith and self.langsmith_api_key:
            self._setup_langsmith()

    def _setup_app_insights(self) -> None:
        """Configure Application Insights tracing."""
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor

            configure_azure_monitor(
                connection_string=self.app_insights_connection_string,
            )
            logger.info("Application Insights tracing enabled")
        except ImportError:
            logger.warning(
                "azure-monitor-opentelemetry not installed. "
                "Install with: pip install azure-monitor-opentelemetry"
            )
        except Exception as e:
            logger.error(f"Failed to configure Application Insights: {e}")

    def _setup_langsmith(self) -> None:
        """Configure LangSmith tracing."""
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
        if self.langsmith_project:
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        logger.info("LangSmith tracing enabled")


@dataclass
class AuthConfig:
    """Authentication configuration."""

    use_managed_identity: bool = True
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    api_key: str = ""

    @classmethod
    def from_env(cls) -> "AuthConfig":
        """Load configuration from environment variables."""
        # Check for managed identity first
        use_mi = os.getenv("USE_MANAGED_IDENTITY", "true").lower() == "true"

        return cls(
            use_managed_identity=use_mi,
            tenant_id=os.getenv("AZURE_TENANT_ID", ""),
            client_id=os.getenv("AZURE_CLIENT_ID", ""),
            client_secret=os.getenv("AZURE_CLIENT_SECRET", ""),
            api_key=os.getenv("API_KEY", ""),
        )

    def get_credential(self):
        """Get the appropriate Azure credential."""
        if self.use_managed_identity:
            try:
                from azure.identity import DefaultAzureCredential

                return DefaultAzureCredential()
            except ImportError:
                logger.warning("azure-identity not installed, falling back to API key")
                return None

        if self.client_id and self.client_secret and self.tenant_id:
            try:
                from azure.identity import ClientSecretCredential

                return ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                )
            except ImportError:
                logger.warning("azure-identity not installed")
                return None

        return None


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    debug: bool = False
    reload: bool = False

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        origins = os.getenv("CORS_ORIGINS", "*").split(",")
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            cors_origins=origins,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            reload=os.getenv("RELOAD", "false").lower() == "true",
        )


@dataclass
class AppConfig:
    """Complete application configuration."""

    azure_openai: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)
    azure_foundry: AzureAIFoundryConfig = field(default_factory=AzureAIFoundryConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # Feature flags
    use_azure_foundry: bool = True
    enable_it_agents: bool = True
    enable_enterprise_agents: bool = True
    enable_deep_agents: bool = True

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load complete configuration from environment variables."""
        config = cls(
            azure_openai=AzureOpenAIConfig.from_env(),
            azure_foundry=AzureAIFoundryConfig.from_env(),
            observability=ObservabilityConfig.from_env(),
            auth=AuthConfig.from_env(),
            server=ServerConfig.from_env(),
            use_azure_foundry=os.getenv("USE_AZURE_FOUNDRY", "true").lower() == "true",
            enable_it_agents=os.getenv("ENABLE_IT_AGENTS", "true").lower() == "true",
            enable_enterprise_agents=os.getenv("ENABLE_ENTERPRISE_AGENTS", "true").lower()
            == "true",
            enable_deep_agents=os.getenv("ENABLE_DEEP_AGENTS", "true").lower() == "true",
        )

        # Validate configuration
        config.validate()

        return config

    def validate(self) -> bool:
        """Validate the complete configuration."""
        is_valid = True

        if not self.azure_openai.validate():
            logger.warning("Azure OpenAI configuration incomplete")
            is_valid = False

        if self.use_azure_foundry and not self.azure_foundry.validate():
            logger.warning("Azure AI Foundry configuration incomplete - using direct mode")
            self.use_azure_foundry = False

        return is_valid

    def initialize(self) -> None:
        """Initialize the application based on configuration."""
        # Configure logging
        self.observability.configure_logging()

        # Set up tracing
        if self.observability.enable_tracing:
            self.observability.setup_tracing()

        logger.info("Application configuration initialized")
        logger.info(f"Azure AI Foundry enabled: {self.use_azure_foundry}")
        logger.info(f"Azure OpenAI endpoint: {self.azure_openai.endpoint}")
        logger.info(f"Tracing enabled: {self.observability.enable_tracing}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding secrets)."""
        return {
            "azure_openai": {
                "endpoint": self.azure_openai.endpoint,
                "deployment_name": self.azure_openai.deployment_name,
                "api_version": self.azure_openai.api_version,
            },
            "azure_foundry": {
                "project_endpoint": self.azure_foundry.project_endpoint,
                "is_enabled": self.azure_foundry.is_enabled,
            },
            "observability": {
                "enable_app_insights": self.observability.enable_app_insights,
                "enable_langsmith": self.observability.enable_langsmith,
                "log_level": self.observability.log_level,
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
            },
            "features": {
                "use_azure_foundry": self.use_azure_foundry,
                "enable_it_agents": self.enable_it_agents,
                "enable_enterprise_agents": self.enable_enterprise_agents,
                "enable_deep_agents": self.enable_deep_agents,
            },
        }


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance.

    Returns:
        The global AppConfig instance.
    """
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None
