"""
LLM Factory
===========

Factory class for creating LLM instances based on configuration.
Supports Ollama, Groq Cloud, OpenAI, Google Gemini, and Hugging Face providers with automatic fallbacks.
"""

from typing import Union, Optional, Any
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from core.llm_config import (
    get_llm_config,
    LLMProvider,
    OllamaConfig,
    GroqConfig,
    OpenAIConfig,
    GoogleGeminiConfig,
    HuggingFaceConfig,
    llm_config_manager,
)
from core.custom_logger import CustomLogger

logger = CustomLogger().get_logger("llm_factory")


class LLMFactory:
    """Factory for creating LLM instances"""

    @staticmethod
    def create_llm(
        force_provider: Optional[LLMProvider] = None, **kwargs
    ) -> Union[OllamaLLM, ChatGroq, ChatOpenAI, ChatGoogleGenerativeAI, Any]:
        """
        Create an LLM instance based on configuration

        Args:
            force_provider: Optional provider to force (overrides config)
            **kwargs: Additional parameters to pass to the LLM

        Returns:
            Configured LLM instance

        Raises:
            ValueError: If configuration is invalid
            ConnectionError: If connection to provider fails
        """
        config_manager = get_llm_config()

        # Use forced provider or default from config
        provider = force_provider or config_manager.provider

        # Validate configuration
        if not config_manager.validate_configuration():
            raise ValueError(f"Invalid configuration for provider: {provider.value}")

        try:
            if provider == LLMProvider.OLLAMA:
                return LLMFactory._create_ollama_llm(
                    config_manager.ollama_config, **kwargs
                )
            elif provider == LLMProvider.GROQ_CLOUD:
                return LLMFactory._create_groq_llm(config_manager.groq_config, **kwargs)
            elif provider == LLMProvider.OPENAI:
                return LLMFactory._create_openai_llm(
                    config_manager.openai_config, **kwargs
                )
            elif provider == LLMProvider.GOOGLE_GEMINI:
                return LLMFactory._create_google_llm(
                    config_manager.google_config, **kwargs
                )
            elif provider == LLMProvider.HUGGINGFACE:
                return LLMFactory._create_huggingface_llm(
                    config_manager.huggingface_config, **kwargs
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            logger.error(f"Failed to create LLM with {provider.value}: {e}")

            # Try fallback if primary fails
            if force_provider is None:
                return LLMFactory._create_fallback_llm(provider, **kwargs)
            else:
                raise

    @staticmethod
    def _create_ollama_llm(config: OllamaConfig, **kwargs) -> OllamaLLM:
        """Create Ollama LLM instance"""
        # Merge config parameters with any overrides
        params = config.to_langchain_params()
        params.update(kwargs)

        try:
            # Try primary model first
            llm = OllamaLLM(**params)
            logger.info(f"Created Ollama LLM with model: {config.primary_model}")
            return llm

        except Exception as e:
            logger.warning(f"Primary model {config.primary_model} failed: {e}")

            # Try backup model
            try:
                params["model"] = config.backup_model
                llm = OllamaLLM(**params)
                logger.info(
                    f"Created Ollama LLM with backup model: {config.backup_model}"
                )
                return llm

            except Exception as e2:
                logger.warning(f"Backup model {config.backup_model} failed: {e2}")

                # Try fallback model
                params["model"] = config.fallback_model
                # Remove advanced parameters for fallback
                fallback_params = {
                    "model": config.fallback_model,
                    "base_url": config.api_url,
                    "temperature": config.temperature,
                    "timeout": config.response_timeout,
                }
                fallback_params.update(kwargs)

                llm = OllamaLLM(**fallback_params)
                logger.info(
                    f"Created Ollama LLM with fallback model: {config.fallback_model}"
                )
                return llm

    @staticmethod
    def _create_groq_llm(config: GroqConfig, **kwargs) -> ChatGroq:
        """Create Groq LLM instance"""
        # Merge config parameters with any overrides
        params = config.to_langchain_params()
        params.update(kwargs)

        try:
            # Try primary model with current API key
            llm = ChatGroq(**params)
            logger.info(f"Created Groq LLM with model: {config.primary_model}")
            return llm

        except Exception as e:
            logger.warning(f"Primary model {config.primary_model} failed: {e}")

            # Try rotating API key if available
            if config.rotate_api_key():
                try:
                    params = config.to_langchain_params()
                    params.update(kwargs)
                    llm = ChatGroq(**params)
                    logger.info(f"Created Groq LLM with rotated API key")
                    return llm
                except Exception as e2:
                    logger.warning(f"Rotated API key failed: {e2}")

            # Try backup model
            try:
                params["model"] = config.backup_model
                llm = ChatGroq(**params)
                logger.info(
                    f"Created Groq LLM with backup model: {config.backup_model}"
                )
                return llm

            except Exception as e3:
                logger.error(f"All Groq models failed: {e3}")
                raise

    @staticmethod
    def _create_openai_llm(config: OpenAIConfig, **kwargs) -> ChatOpenAI:
        """Create OpenAI LLM instance"""
        # Merge config parameters with any overrides
        params = config.to_langchain_params()
        params.update(kwargs)

        try:
            # Try primary model first
            llm = ChatOpenAI(**params)
            logger.info(f"Created OpenAI LLM with model: {config.primary_model}")
            return llm

        except Exception as e:
            logger.warning(f"Primary model {config.primary_model} failed: {e}")

            # Try rotating API key if available
            if config.rotate_api_key():
                try:
                    params = config.to_langchain_params()
                    params.update(kwargs)
                    llm = ChatOpenAI(**params)
                    logger.info(f"Created OpenAI LLM with rotated API key")
                    return llm
                except Exception as e2:
                    logger.warning(f"Rotated API key failed: {e2}")

            # Try backup model
            try:
                params["model"] = config.backup_model
                llm = ChatOpenAI(**params)
                logger.info(
                    f"Created OpenAI LLM with backup model: {config.backup_model}"
                )
                return llm

            except Exception as e3:
                logger.error(f"All OpenAI models failed: {e3}")
                raise

    @staticmethod
    def _create_google_llm(
        config: GoogleGeminiConfig, **kwargs
    ) -> ChatGoogleGenerativeAI:
        """Create Google Gemini LLM instance"""
        # Merge config parameters with any overrides
        params = config.to_langchain_params()
        params.update(kwargs)

        try:
            # Try primary model first
            llm = ChatGoogleGenerativeAI(**params)
            logger.info(f"Created Google Gemini LLM with model: {config.primary_model}")
            return llm

        except Exception as e:
            logger.warning(f"Primary model {config.primary_model} failed: {e}")

            # Try rotating API key if available
            if config.rotate_api_key():
                try:
                    params = config.to_langchain_params()
                    params.update(kwargs)
                    llm = ChatGoogleGenerativeAI(**params)
                    logger.info(f"Created Google Gemini LLM with rotated API key")
                    return llm
                except Exception as e2:
                    logger.warning(f"Rotated API key failed: {e2}")

            # Try backup model
            try:
                params["model"] = config.backup_model
                llm = ChatGoogleGenerativeAI(**params)
                logger.info(
                    f"Created Google Gemini LLM with backup model: {config.backup_model}"
                )
                return llm

            except Exception as e3:
                logger.error(f"All Google Gemini models failed: {e3}")
                raise

    @staticmethod
    def _create_huggingface_llm(config: HuggingFaceConfig, **kwargs):
        """Create Hugging Face LLM instance"""
        try:
            # Import here to avoid dependency issues if not installed
            from langchain_huggingface import HuggingFacePipeline
        except ImportError:
            raise ValueError(
                "langchain_huggingface not installed. Please install it with: pip install langchain-huggingface"
            )

        # Merge config parameters with any overrides
        params = config.to_langchain_params()
        params.update(kwargs)

        try:
            llm = HuggingFacePipeline.from_model_id(**params)
            logger.info(f"Created Hugging Face LLM with model: {config.model_id}")
            return llm

        except Exception as e:
            logger.error(f"Hugging Face model {config.model_id} failed: {e}")
            raise

    @staticmethod
    def _create_fallback_llm(
        failed_provider: LLMProvider, **kwargs
    ) -> Union[OllamaLLM, ChatGroq, ChatOpenAI, ChatGoogleGenerativeAI, Any]:
        """Create fallback LLM when primary provider fails"""
        config_manager = get_llm_config()

        # Define fallback priority order
        fallback_order = {
            LLMProvider.OLLAMA: [
                LLMProvider.HUGGINGFACE,
                LLMProvider.GROQ_CLOUD,
                LLMProvider.OPENAI,
            ],
            LLMProvider.GROQ_CLOUD: [
                LLMProvider.OLLAMA,
                LLMProvider.OPENAI,
                LLMProvider.GOOGLE_GEMINI,
            ],
            LLMProvider.OPENAI: [
                LLMProvider.GROQ_CLOUD,
                LLMProvider.GOOGLE_GEMINI,
                LLMProvider.OLLAMA,
            ],
            LLMProvider.GOOGLE_GEMINI: [
                LLMProvider.OPENAI,
                LLMProvider.GROQ_CLOUD,
                LLMProvider.OLLAMA,
            ],
            LLMProvider.HUGGINGFACE: [
                LLMProvider.OLLAMA,
                LLMProvider.GROQ_CLOUD,
                LLMProvider.OPENAI,
            ],
        }

        fallback_providers = fallback_order.get(failed_provider, [])

        for fallback_provider in fallback_providers:
            try:
                logger.info(
                    f"Attempting {fallback_provider.value} as fallback for failed {failed_provider.value}"
                )

                # Validate fallback provider
                temp_provider = config_manager.provider
                config_manager.provider = fallback_provider
                is_valid = config_manager.validate_configuration()
                config_manager.provider = temp_provider

                if not is_valid:
                    logger.warning(
                        f"{fallback_provider.value} configuration is invalid, skipping"
                    )
                    continue

                # Create fallback LLM
                if fallback_provider == LLMProvider.OLLAMA:
                    return LLMFactory._create_ollama_llm(
                        config_manager.ollama_config, **kwargs
                    )
                elif fallback_provider == LLMProvider.GROQ_CLOUD:
                    return LLMFactory._create_groq_llm(
                        config_manager.groq_config, **kwargs
                    )
                elif fallback_provider == LLMProvider.OPENAI:
                    return LLMFactory._create_openai_llm(
                        config_manager.openai_config, **kwargs
                    )
                elif fallback_provider == LLMProvider.GOOGLE_GEMINI:
                    return LLMFactory._create_google_llm(
                        config_manager.google_config, **kwargs
                    )
                elif fallback_provider == LLMProvider.HUGGINGFACE:
                    return LLMFactory._create_huggingface_llm(
                        config_manager.huggingface_config, **kwargs
                    )

            except Exception as e:
                logger.error(f"{fallback_provider.value} fallback failed: {e}")
                continue

        raise ConnectionError(f"All LLM providers failed, no fallback available")

    @staticmethod
    def create_chat_llm(**kwargs) -> Union[ChatOllama, ChatGroq]:
        """
        Create a chat-oriented LLM instance
        For applications that need conversational capabilities
        """
        config_manager = get_llm_config()

        if config_manager.is_ollama_enabled():
            # Use ChatOllama for conversation
            config = config_manager.ollama_config
            params = {
                "model": config.primary_model,
                "base_url": config.api_url,
                "temperature": config.temperature,
                "timeout": config.response_timeout,
            }
            params.update(kwargs)

            try:
                llm = ChatOllama(**params)
                logger.info(f"Created ChatOllama with model: {config.primary_model}")
                return llm
            except Exception as e:
                logger.error(f"Failed to create ChatOllama: {e}")
                raise

        else:
            # Use ChatGroq
            return LLMFactory.create_llm(**kwargs)

    @staticmethod
    def test_llm_connection(provider: Optional[LLMProvider] = None) -> bool:
        """
        Test LLM connection and basic functionality

        Args:
            provider: Optional provider to test, defaults to configured provider

        Returns:
            True if connection successful, False otherwise
        """
        try:
            llm = LLMFactory.create_llm(force_provider=provider)

            # Test with a simple prompt
            test_prompt = "Hello, respond with 'OK' if you can understand this."

            if hasattr(llm, "invoke"):
                response = llm.invoke(test_prompt)
            else:
                response = llm(test_prompt)

            logger.info(f"LLM test successful: {response[:50]}...")
            return True

        except Exception as e:
            logger.error(f"LLM test failed: {e}")
            return False

    @staticmethod
    def get_provider_status() -> dict:
        """Get status of all providers"""
        config_manager = get_llm_config()

        status = {
            "current_provider": config_manager.provider.value,
            "ollama": {"available": False, "models": [], "error": None},
            "groq": {
                "available": False,
                "api_keys_count": len(config_manager.groq_config.api_keys),
                "error": None,
            },
        }

        # Test Ollama
        try:
            if config_manager._validate_ollama():
                status["ollama"]["available"] = True
                # Try to get available models
                try:
                    import requests

                    response = requests.get(
                        f"{config_manager.ollama_config.api_url}/api/tags", timeout=5
                    )
                    if response.status_code == 200:
                        models_data = response.json()
                        status["ollama"]["models"] = [
                            model["name"] for model in models_data.get("models", [])
                        ]
                except Exception as e:
                    status["ollama"]["error"] = str(e)
        except Exception as e:
            status["ollama"]["error"] = str(e)

        # Test Groq
        try:
            if config_manager._validate_groq():
                status["groq"]["available"] = True
        except Exception as e:
            status["groq"]["error"] = str(e)

        return status


# Convenience functions
def create_llm(**kwargs) -> Union[OllamaLLM, ChatGroq]:
    """Create LLM with current configuration"""
    return LLMFactory.create_llm(**kwargs)


def create_chat_llm(**kwargs) -> Union[ChatOllama, ChatGroq]:
    """Create chat LLM with current configuration"""
    return LLMFactory.create_chat_llm(**kwargs)


def test_llm_connection() -> bool:
    """Test current LLM connection"""
    return LLMFactory.test_llm_connection()


def switch_provider(provider_name: str) -> bool:
    """Switch LLM provider by name"""
    config_manager = get_llm_config()

    if provider_name.lower() == "ollama":
        return config_manager.switch_provider(LLMProvider.OLLAMA)
    elif provider_name.lower() in ["groq", "groq_cloud"]:
        return config_manager.switch_provider(LLMProvider.GROQ_CLOUD)
    else:
        logger.error(f"Unknown provider: {provider_name}")
        return False


def get_current_provider() -> str:
    """Get current provider name"""
    return get_llm_config().provider.value
