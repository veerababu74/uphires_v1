"""
LLM Configuration Management
===========================

This module provides configuration management for different LLM providers:
- Ollama (Local)
- Groq Cloud (API-based)

Users can configure which LLM provider to use and all associated settings.
"""

import os
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from dotenv import load_dotenv
from dataclasses import dataclass
from core.custom_logger import CustomLogger

# Load environment variables
load_dotenv()

logger = CustomLogger().get_logger("llm_config")


class LLMProvider(Enum):
    """Supported LLM providers"""

    OLLAMA = "ollama"
    GROQ_CLOUD = "groq_cloud"


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM"""

    # Connection Settings
    api_url: str = "http://localhost:11434"
    connection_timeout: int = 5
    request_timeout: int = 60

    # Model Settings
    primary_model: str = "llama3.2:3b"
    backup_model: str = "qwen2.5:3b"
    fallback_model: str = "qwen:4b"

    # Generation Parameters
    temperature: float = 0.1
    num_predict: int = 1024
    top_k: int = 20
    top_p: float = 0.8
    repeat_penalty: float = 1.1
    response_timeout: int = 30

    # Processing Settings
    format: str = "json"  # Force JSON output
    max_context_length: int = 8000
    enable_debug: bool = True
    enable_fallback: bool = True

    # Advanced Settings
    stream: bool = False
    raw: bool = False
    keep_alive: str = "5m"

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Create OllamaConfig from environment variables"""
        return cls(
            api_url=os.getenv("OLLAMA_API_URL", cls.api_url),
            connection_timeout=int(
                os.getenv("OLLAMA_CONNECTION_TIMEOUT", cls.connection_timeout)
            ),
            request_timeout=int(
                os.getenv("OLLAMA_REQUEST_TIMEOUT", cls.request_timeout)
            ),
            primary_model=os.getenv("OLLAMA_PRIMARY_MODEL", cls.primary_model),
            backup_model=os.getenv("OLLAMA_BACKUP_MODEL", cls.backup_model),
            fallback_model=os.getenv("OLLAMA_FALLBACK_MODEL", cls.fallback_model),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", cls.temperature)),
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", cls.num_predict)),
            top_k=int(os.getenv("OLLAMA_TOP_K", cls.top_k)),
            top_p=float(os.getenv("OLLAMA_TOP_P", cls.top_p)),
            repeat_penalty=float(
                os.getenv("OLLAMA_REPEAT_PENALTY", cls.repeat_penalty)
            ),
            response_timeout=int(
                os.getenv("OLLAMA_RESPONSE_TIMEOUT", cls.response_timeout)
            ),
            max_context_length=int(
                os.getenv("OLLAMA_MAX_CONTEXT_LENGTH", cls.max_context_length)
            ),
            enable_debug=os.getenv("OLLAMA_ENABLE_DEBUG", "true").lower() == "true",
            enable_fallback=os.getenv("OLLAMA_ENABLE_FALLBACK", "true").lower()
            == "true",
        )

    def to_langchain_params(self) -> Dict[str, Any]:
        """Convert to LangChain OllamaLLM parameters"""
        return {
            "model": self.primary_model,
            "base_url": self.api_url,
            "temperature": self.temperature,
            "format": self.format,
            "num_predict": self.num_predict,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "timeout": self.response_timeout,
            "keep_alive": self.keep_alive,
        }


@dataclass
class GroqConfig:
    """Configuration for Groq Cloud LLM"""

    # API Settings
    api_keys: List[str] = None
    current_key_index: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Model Settings
    primary_model: str = "gemma2-9b-it"
    backup_model: str = "llama-3.1-70b-versatile"
    fallback_model: str = "mixtral-8x7b-32768"

    # Generation Parameters
    temperature: float = 0.1
    max_tokens: int = 1024
    top_p: float = 0.8
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Request Settings
    request_timeout: int = 60
    max_context_length: int = 8000
    enable_streaming: bool = False

    # Rate Limiting
    requests_per_minute: int = 30
    tokens_per_minute: int = 6000

    def __post_init__(self):
        if self.api_keys is None:
            self.api_keys = self._get_api_keys_from_env()

    @staticmethod
    def _get_api_keys_from_env() -> List[str]:
        """Get API keys from environment variables"""
        api_keys_str = os.getenv("GROQ_API_KEYS", "")
        if not api_keys_str:
            return []
        return [key.strip() for key in api_keys_str.split(",") if key.strip()]

    @classmethod
    def from_env(cls) -> "GroqConfig":
        """Create GroqConfig from environment variables"""
        return cls(
            api_keys=cls._get_api_keys_from_env(),
            max_retries=int(os.getenv("GROQ_MAX_RETRIES", cls.max_retries)),
            retry_delay=float(os.getenv("GROQ_RETRY_DELAY", cls.retry_delay)),
            primary_model=os.getenv("GROQ_PRIMARY_MODEL", cls.primary_model),
            backup_model=os.getenv("GROQ_BACKUP_MODEL", cls.backup_model),
            fallback_model=os.getenv("GROQ_FALLBACK_MODEL", cls.fallback_model),
            temperature=float(os.getenv("GROQ_TEMPERATURE", cls.temperature)),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", cls.max_tokens)),
            top_p=float(os.getenv("GROQ_TOP_P", cls.top_p)),
            request_timeout=int(os.getenv("GROQ_REQUEST_TIMEOUT", cls.request_timeout)),
            max_context_length=int(
                os.getenv("GROQ_MAX_CONTEXT_LENGTH", cls.max_context_length)
            ),
            requests_per_minute=int(
                os.getenv("GROQ_REQUESTS_PER_MINUTE", cls.requests_per_minute)
            ),
            tokens_per_minute=int(
                os.getenv("GROQ_TOKENS_PER_MINUTE", cls.tokens_per_minute)
            ),
        )

    def get_current_api_key(self) -> Optional[str]:
        """Get the current API key"""
        if not self.api_keys or self.current_key_index >= len(self.api_keys):
            return None
        return self.api_keys[self.current_key_index]

    def rotate_api_key(self) -> bool:
        """Rotate to next API key. Returns True if rotation successful."""
        if not self.api_keys or len(self.api_keys) <= 1:
            return False

        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(f"Rotated to API key index: {self.current_key_index}")
        return True

    def to_langchain_params(self) -> Dict[str, Any]:
        """Convert to LangChain ChatGroq parameters"""
        api_key = self.get_current_api_key()
        if not api_key:
            raise ValueError("No valid Groq API key available")

        return {
            "api_key": api_key,
            "model": self.primary_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.request_timeout,
        }


class LLMConfigManager:
    """Centralized LLM configuration manager"""

    def __init__(self):
        self._provider = self._get_provider_from_env()
        self._ollama_config = OllamaConfig.from_env()
        self._groq_config = GroqConfig.from_env()
        self._validated = False

    @staticmethod
    def _get_provider_from_env() -> LLMProvider:
        """Get LLM provider from environment variable"""
        provider_str = os.getenv("LLM_PROVIDER", "ollama").lower()

        if provider_str == "groq_cloud" or provider_str == "groq":
            return LLMProvider.GROQ_CLOUD
        elif provider_str == "ollama":
            return LLMProvider.OLLAMA
        else:
            logger.warning(
                f"Unknown LLM provider '{provider_str}', defaulting to Ollama"
            )
            return LLMProvider.OLLAMA

    @property
    def provider(self) -> LLMProvider:
        """Get current LLM provider"""
        return self._provider

    @provider.setter
    def provider(self, value: LLMProvider):
        """Set LLM provider"""
        self._provider = value
        self._validated = False
        logger.info(f"LLM provider set to: {value.value}")

    @property
    def ollama_config(self) -> OllamaConfig:
        """Get Ollama configuration"""
        return self._ollama_config

    @property
    def groq_config(self) -> GroqConfig:
        """Get Groq configuration"""
        return self._groq_config

    def is_ollama_enabled(self) -> bool:
        """Check if Ollama is the current provider"""
        return self._provider == LLMProvider.OLLAMA

    def is_groq_enabled(self) -> bool:
        """Check if Groq Cloud is the current provider"""
        return self._provider == LLMProvider.GROQ_CLOUD

    def validate_configuration(self) -> bool:
        """Validate the current configuration"""
        if self._validated:
            return True

        try:
            if self.is_ollama_enabled():
                success = self._validate_ollama()
            else:
                success = self._validate_groq()

            self._validated = success
            return success

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def _validate_ollama(self) -> bool:
        """Validate Ollama configuration"""
        try:
            import requests

            # Check Ollama connection
            response = requests.get(
                f"{self.ollama_config.api_url}/api/tags",
                timeout=self.ollama_config.connection_timeout,
            )

            if response.status_code != 200:
                logger.error(
                    f"Ollama server not accessible at {self.ollama_config.api_url}"
                )
                return False

            # Check if primary model is available
            models_data = response.json()
            available_models = [
                model["name"] for model in models_data.get("models", [])
            ]

            if not any(
                self.ollama_config.primary_model in model for model in available_models
            ):
                logger.warning(
                    f"Primary model '{self.ollama_config.primary_model}' not found"
                )

                # Check backup model
                if not any(
                    self.ollama_config.backup_model in model
                    for model in available_models
                ):
                    logger.error(f"Neither primary nor backup model available")
                    return False
                else:
                    logger.info(
                        f"Using backup model: {self.ollama_config.backup_model}"
                    )

            logger.info("Ollama configuration validated successfully")
            return True

        except Exception as e:
            logger.error(f"Ollama validation failed: {e}")
            return False

    def _validate_groq(self) -> bool:
        """Validate Groq configuration"""
        if not self.groq_config.api_keys:
            logger.error("No Groq API keys configured")
            return False

        # Test API key validity (basic check)
        api_key = self.groq_config.get_current_api_key()
        if not api_key or len(api_key.strip()) < 10:
            logger.error("Invalid Groq API key format")
            return False

        logger.info(
            f"Groq configuration validated with {len(self.groq_config.api_keys)} API key(s)"
        )
        return True

    def get_active_config(self) -> Union[OllamaConfig, GroqConfig]:
        """Get the configuration for the active provider"""
        if self.is_ollama_enabled():
            return self.ollama_config
        else:
            return self.groq_config

    def get_langchain_params(self) -> Dict[str, Any]:
        """Get LangChain parameters for the active provider"""
        if self.is_ollama_enabled():
            return self.ollama_config.to_langchain_params()
        else:
            return self.groq_config.to_langchain_params()

    def switch_provider(self, provider: LLMProvider) -> bool:
        """Switch to a different LLM provider"""
        old_provider = self._provider
        self.provider = provider

        if self.validate_configuration():
            logger.info(
                f"Successfully switched from {old_provider.value} to {provider.value}"
            )
            return True
        else:
            # Revert on failure
            self._provider = old_provider
            logger.error(
                f"Failed to switch to {provider.value}, reverted to {old_provider.value}"
            )
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current configuration status"""
        return {
            "provider": self._provider.value,
            "validated": self._validated,
            "ollama_available": (
                self._validate_ollama() if not self._validated else None
            ),
            "groq_keys_count": len(self.groq_config.api_keys),
            "ollama_config": {
                "api_url": self.ollama_config.api_url,
                "primary_model": self.ollama_config.primary_model,
                "backup_model": self.ollama_config.backup_model,
            },
            "groq_config": {
                "primary_model": self.groq_config.primary_model,
                "current_key_index": self.groq_config.current_key_index,
            },
        }


# Global instance
llm_config_manager = LLMConfigManager()


def get_llm_config() -> LLMConfigManager:
    """Get the global LLM configuration manager"""
    return llm_config_manager


def configure_llm_provider(provider: str) -> bool:
    """Configure LLM provider from string"""
    try:
        if provider.lower() in ["ollama"]:
            return llm_config_manager.switch_provider(LLMProvider.OLLAMA)
        elif provider.lower() in ["groq", "groq_cloud"]:
            return llm_config_manager.switch_provider(LLMProvider.GROQ_CLOUD)
        else:
            logger.error(f"Unknown provider: {provider}")
            return False
    except Exception as e:
        logger.error(f"Error configuring provider: {e}")
        return False


# Convenience functions for backward compatibility
def is_ollama_enabled() -> bool:
    """Check if Ollama is enabled"""
    return llm_config_manager.is_ollama_enabled()


def is_groq_enabled() -> bool:
    """Check if Groq is enabled"""
    return llm_config_manager.is_groq_enabled()


def get_ollama_config() -> OllamaConfig:
    """Get Ollama configuration"""
    return llm_config_manager.ollama_config


def get_groq_config() -> GroqConfig:
    """Get Groq configuration"""
    return llm_config_manager.groq_config
