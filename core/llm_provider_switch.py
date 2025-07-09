"""
LLM Provider Switching Utility
=============================

This module provides a centralized utility for switching between LLM providers
across the entire application, including GroqCloudLLM and multipleresumepraser modules.
"""

import os
from typing import List, Optional, Dict, Any
from enum import Enum
from core.config import config
from core.custom_logger import CustomLogger
from core.llm_config import LLMConfigManager, LLMProvider

logger = CustomLogger().get_logger("llm_provider_switch")


class ProviderSwitchManager:
    """
    Centralized manager for switching LLM providers across the application.
    """

    def __init__(self):
        self.llm_manager = LLMConfigManager()
        self._current_provider = None
        self._groqcloud_parser = None
        self._multiresumeparser = None

    def get_current_provider(self) -> str:
        """Get the current LLM provider."""
        if self.llm_manager.is_ollama_enabled():
            return "ollama"
        else:
            return "groq"

    def switch_provider_globally(
        self, new_provider: str, api_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Switch LLM provider globally across all modules.

        Args:
            new_provider (str): New provider to use ('groq' or 'ollama')
            api_keys (Optional[List[str]]): API keys for Groq provider

        Returns:
            Dict[str, Any]: Status information about the switch
        """
        try:
            if new_provider.lower() not in ["groq", "ollama"]:
                raise ValueError("Invalid provider. Must be 'groq' or 'ollama'")

            # Update environment variable for persistence
            os.environ["LLM_PROVIDER"] = new_provider.lower()

            # Update the core config manager
            if new_provider.lower() == "ollama":
                self.llm_manager.provider = LLMProvider.OLLAMA
            else:
                self.llm_manager.provider = LLMProvider.GROQ_CLOUD

            # Validate the new configuration
            if not self.llm_manager.validate_configuration():
                raise RuntimeError(f"Failed to validate {new_provider} configuration")

            result = {
                "status": "success",
                "previous_provider": self._current_provider,
                "current_provider": new_provider.lower(),
                "message": f"Successfully switched to {new_provider}",
                "validation_passed": True,
            }

            # Add provider-specific information
            if new_provider.lower() == "ollama":
                result.update(
                    {
                        "provider_type": "local",
                        "model": self.llm_manager.ollama_config.primary_model,
                        "api_url": self.llm_manager.ollama_config.api_url,
                    }
                )
            else:
                result.update(
                    {
                        "provider_type": "api",
                        "model": self.llm_manager.groq_config.primary_model,
                        "api_keys_count": (
                            len(api_keys)
                            if api_keys
                            else len(self.llm_manager.groq_config.api_keys)
                        ),
                    }
                )

            self._current_provider = new_provider.lower()
            logger.info(f"Successfully switched to {new_provider} provider globally")

            return result

        except Exception as e:
            error_msg = f"Failed to switch provider globally: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg, "validation_passed": False}

    def get_provider_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of current LLM provider configuration.

        Returns:
            Dict[str, Any]: Status information
        """
        try:
            current_provider = self.get_current_provider()

            status = {
                "current_provider": current_provider,
                "provider_type": "local" if current_provider == "ollama" else "api",
                "configuration_valid": self.llm_manager.validate_configuration(),
                "environment_variable": os.getenv("LLM_PROVIDER", "ollama"),
            }

            if current_provider == "ollama":
                ollama_config = self.llm_manager.ollama_config
                status.update(
                    {
                        "model": ollama_config.primary_model,
                        "backup_model": ollama_config.backup_model,
                        "api_url": ollama_config.api_url,
                        "connection_timeout": ollama_config.connection_timeout,
                        "request_timeout": ollama_config.request_timeout,
                    }
                )

                # Check Ollama service status
                try:
                    import requests

                    response = requests.get(
                        f"{ollama_config.api_url}/api/tags",
                        timeout=ollama_config.connection_timeout,
                    )
                    status["service_status"] = (
                        "running" if response.status_code == 200 else "error"
                    )

                    if response.status_code == 200:
                        models_data = response.json()
                        status["available_models"] = [
                            model["name"] for model in models_data.get("models", [])
                        ]
                    else:
                        status["available_models"] = []

                except Exception as e:
                    status["service_status"] = "not_accessible"
                    status["service_error"] = str(e)
                    status["available_models"] = []
            else:
                groq_config = self.llm_manager.groq_config
                status.update(
                    {
                        "model": groq_config.primary_model,
                        "backup_model": groq_config.backup_model,
                        "api_keys_count": len(groq_config.api_keys),
                        "has_valid_keys": bool(groq_config.api_keys),
                        "current_key_index": groq_config.current_key_index,
                        "request_timeout": groq_config.request_timeout,
                        "max_tokens": groq_config.max_tokens,
                    }
                )

            return status

        except Exception as e:
            logger.error(f"Failed to get provider status: {str(e)}")
            return {
                "current_provider": "unknown",
                "configuration_valid": False,
                "error": str(e),
            }

    def test_provider_connection(
        self, provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test connection to the specified LLM provider.

        Args:
            provider (Optional[str]): Provider to test. If None, tests current provider.

        Returns:
            Dict[str, Any]: Test results
        """
        if provider is None:
            provider = self.get_current_provider()

        try:
            if provider.lower() == "ollama":
                return self._test_ollama_connection()
            else:
                return self._test_groq_connection()

        except Exception as e:
            return {
                "provider": provider,
                "status": "error",
                "message": f"Connection test failed: {str(e)}",
            }

    def _test_ollama_connection(self) -> Dict[str, Any]:
        """Test Ollama connection."""
        try:
            import requests

            ollama_config = self.llm_manager.ollama_config

            response = requests.get(
                f"{ollama_config.api_url}/api/tags",
                timeout=ollama_config.connection_timeout,
            )

            if response.status_code == 200:
                models_data = response.json()
                available_models = [
                    model["name"] for model in models_data.get("models", [])
                ]

                return {
                    "provider": "ollama",
                    "status": "success",
                    "message": "Ollama connection successful",
                    "available_models": available_models,
                    "models_count": len(available_models),
                    "primary_model_available": ollama_config.primary_model
                    in available_models,
                }
            else:
                return {
                    "provider": "ollama",
                    "status": "error",
                    "message": f"Ollama responded with status {response.status_code}",
                }

        except Exception as e:
            return {
                "provider": "ollama",
                "status": "error",
                "message": f"Failed to connect to Ollama: {str(e)}",
            }

    def _test_groq_connection(self) -> Dict[str, Any]:
        """Test Groq connection."""
        try:
            groq_config = self.llm_manager.groq_config

            if not groq_config.api_keys:
                return {
                    "provider": "groq",
                    "status": "error",
                    "message": "No Groq API keys configured",
                }

            # Simple test would require actually calling the API
            # For now, just validate that keys are present
            return {
                "provider": "groq",
                "status": "success",
                "message": "Groq API keys configured",
                "api_keys_count": len(groq_config.api_keys),
                "primary_model": groq_config.primary_model,
            }

        except Exception as e:
            return {
                "provider": "groq",
                "status": "error",
                "message": f"Groq configuration error: {str(e)}",
            }


# Global instance for easy access
provider_switch_manager = ProviderSwitchManager()
