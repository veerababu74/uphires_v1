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
    OPENAI = "openai"
    GOOGLE_GEMINI = "google_gemini"
    HUGGINGFACE = "huggingface"


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
            "timeout": self.request_timeout,
            "model_kwargs": {
                "top_p": self.top_p,
            },
        }


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI LLM"""

    # API Settings
    api_keys: List[str] = None
    current_key_index: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    organization: Optional[str] = None

    # Model Settings
    primary_model: str = "gpt-3.5-turbo"
    backup_model: str = "gpt-3.5-turbo-instruct"
    fallback_model: str = "gpt-3.5-turbo"

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
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000

    def __post_init__(self):
        if self.api_keys is None:
            self.api_keys = self._get_api_keys_from_env()

    @staticmethod
    def _get_api_keys_from_env() -> List[str]:
        """Get API keys from environment variables"""
        api_keys_str = os.getenv("OPENAI_API_KEYS", "")
        if not api_keys_str:
            # Try single key format
            single_key = os.getenv("OPENAI_API_KEY", "")
            if single_key:
                return [single_key]
            return []
        return [key.strip() for key in api_keys_str.split(",") if key.strip()]

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Create OpenAIConfig from environment variables"""
        return cls(
            api_keys=cls._get_api_keys_from_env(),
            organization=os.getenv("OPENAI_ORGANIZATION"),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", cls.max_retries)),
            retry_delay=float(os.getenv("OPENAI_RETRY_DELAY", cls.retry_delay)),
            primary_model=os.getenv("OPENAI_PRIMARY_MODEL", cls.primary_model),
            backup_model=os.getenv("OPENAI_BACKUP_MODEL", cls.backup_model),
            fallback_model=os.getenv("OPENAI_FALLBACK_MODEL", cls.fallback_model),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", cls.temperature)),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", cls.max_tokens)),
            top_p=float(os.getenv("OPENAI_TOP_P", cls.top_p)),
            frequency_penalty=float(
                os.getenv("OPENAI_FREQUENCY_PENALTY", cls.frequency_penalty)
            ),
            presence_penalty=float(
                os.getenv("OPENAI_PRESENCE_PENALTY", cls.presence_penalty)
            ),
            request_timeout=int(
                os.getenv("OPENAI_REQUEST_TIMEOUT", cls.request_timeout)
            ),
            max_context_length=int(
                os.getenv("OPENAI_MAX_CONTEXT_LENGTH", cls.max_context_length)
            ),
            requests_per_minute=int(
                os.getenv("OPENAI_REQUESTS_PER_MINUTE", cls.requests_per_minute)
            ),
            tokens_per_minute=int(
                os.getenv("OPENAI_TOKENS_PER_MINUTE", cls.tokens_per_minute)
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
        logger.info(f"Rotated to OpenAI API key index: {self.current_key_index}")
        return True

    def to_langchain_params(self) -> Dict[str, Any]:
        """Convert to LangChain ChatOpenAI parameters"""
        api_key = self.get_current_api_key()
        if not api_key:
            raise ValueError("No valid OpenAI API key available")

        params = {
            "api_key": api_key,
            "model": self.primary_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "timeout": self.request_timeout,
        }

        if self.organization:
            params["organization"] = self.organization

        return params


@dataclass
class GoogleGeminiConfig:
    """Configuration for Google Gemini LLM"""

    # API Settings
    api_keys: List[str] = None
    current_key_index: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Model Settings
    primary_model: str = "gemini-1.5-flash"
    backup_model: str = "gemini-1.5-pro"
    fallback_model: str = "gemini-pro"

    # Generation Parameters
    temperature: float = 0.1
    max_tokens: int = 1024
    top_p: float = 0.8
    top_k: int = 40

    # Request Settings
    request_timeout: int = 60
    max_context_length: int = 8000
    enable_streaming: bool = False

    # Rate Limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 32000

    def __post_init__(self):
        if self.api_keys is None:
            self.api_keys = self._get_api_keys_from_env()

    @staticmethod
    def _get_api_keys_from_env() -> List[str]:
        """Get API keys from environment variables"""
        api_keys_str = os.getenv("GOOGLE_API_KEYS", "")
        if not api_keys_str:
            # Try single key format
            single_key = os.getenv("GOOGLE_API_KEY", "")
            if single_key:
                return [single_key]
            return []
        return [key.strip() for key in api_keys_str.split(",") if key.strip()]

    @classmethod
    def from_env(cls) -> "GoogleGeminiConfig":
        """Create GoogleGeminiConfig from environment variables"""
        return cls(
            api_keys=cls._get_api_keys_from_env(),
            max_retries=int(os.getenv("GOOGLE_MAX_RETRIES", cls.max_retries)),
            retry_delay=float(os.getenv("GOOGLE_RETRY_DELAY", cls.retry_delay)),
            primary_model=os.getenv("GOOGLE_PRIMARY_MODEL", cls.primary_model),
            backup_model=os.getenv("GOOGLE_BACKUP_MODEL", cls.backup_model),
            fallback_model=os.getenv("GOOGLE_FALLBACK_MODEL", cls.fallback_model),
            temperature=float(os.getenv("GOOGLE_TEMPERATURE", cls.temperature)),
            max_tokens=int(os.getenv("GOOGLE_MAX_TOKENS", cls.max_tokens)),
            top_p=float(os.getenv("GOOGLE_TOP_P", cls.top_p)),
            top_k=int(os.getenv("GOOGLE_TOP_K", cls.top_k)),
            request_timeout=int(
                os.getenv("GOOGLE_REQUEST_TIMEOUT", cls.request_timeout)
            ),
            max_context_length=int(
                os.getenv("GOOGLE_MAX_CONTEXT_LENGTH", cls.max_context_length)
            ),
            requests_per_minute=int(
                os.getenv("GOOGLE_REQUESTS_PER_MINUTE", cls.requests_per_minute)
            ),
            tokens_per_minute=int(
                os.getenv("GOOGLE_TOKENS_PER_MINUTE", cls.tokens_per_minute)
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
        logger.info(f"Rotated to Google API key index: {self.current_key_index}")
        return True

    def to_langchain_params(self) -> Dict[str, Any]:
        """Convert to LangChain ChatGoogleGenerativeAI parameters"""
        api_key = self.get_current_api_key()
        if not api_key:
            raise ValueError("No valid Google API key available")

        return {
            "google_api_key": api_key,
            "model": self.primary_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "timeout": self.request_timeout,
        }


@dataclass
class HuggingFaceConfig:
    """Configuration for Hugging Face LLM"""

    # Model Settings
    model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    task: str = "text-generation"
    device: Optional[str] = None  # Auto-detect if None
    model_kwargs: Optional[Dict[str, Any]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None

    # Pipeline Parameters
    max_new_tokens: int = 1024
    temperature: float = 0.1
    top_k: int = 50
    top_p: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.1

    # Processing Settings
    max_context_length: int = 4000
    batch_size: int = 1
    trust_remote_code: bool = False

    # Authentication (for private models)
    hf_token: Optional[str] = None

    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {}
        if self.hf_token is None:
            self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

    @classmethod
    def from_env(cls) -> "HuggingFaceConfig":
        """Create HuggingFaceConfig from environment variables"""
        return cls(
            model_id=os.getenv("HUGGINGFACE_MODEL_ID", cls.model_id),
            task=os.getenv("HUGGINGFACE_TASK", cls.task),
            device=os.getenv("HUGGINGFACE_DEVICE"),
            max_new_tokens=int(
                os.getenv("HUGGINGFACE_MAX_NEW_TOKENS", cls.max_new_tokens)
            ),
            temperature=float(os.getenv("HUGGINGFACE_TEMPERATURE", cls.temperature)),
            top_k=int(os.getenv("HUGGINGFACE_TOP_K", cls.top_k)),
            top_p=float(os.getenv("HUGGINGFACE_TOP_P", cls.top_p)),
            repetition_penalty=float(
                os.getenv("HUGGINGFACE_REPETITION_PENALTY", cls.repetition_penalty)
            ),
            max_context_length=int(
                os.getenv("HUGGINGFACE_MAX_CONTEXT_LENGTH", cls.max_context_length)
            ),
            batch_size=int(os.getenv("HUGGINGFACE_BATCH_SIZE", cls.batch_size)),
            trust_remote_code=os.getenv(
                "HUGGINGFACE_TRUST_REMOTE_CODE", "false"
            ).lower()
            == "true",
            hf_token=os.getenv("HUGGINGFACE_TOKEN"),
        )

    def to_langchain_params(self) -> Dict[str, Any]:
        """Convert to LangChain HuggingFacePipeline parameters"""
        pipeline_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }

        params = {
            "model_id": self.model_id,
            "task": self.task,
            "pipeline_kwargs": pipeline_kwargs,
            "model_kwargs": self.model_kwargs.copy(),
            "tokenizer_kwargs": self.tokenizer_kwargs.copy(),
        }

        if self.device:
            params["device"] = self.device

        if self.hf_token:
            params["model_kwargs"]["token"] = self.hf_token
            params["tokenizer_kwargs"]["token"] = self.hf_token

        if self.trust_remote_code:
            params["model_kwargs"]["trust_remote_code"] = True
            params["tokenizer_kwargs"]["trust_remote_code"] = True

        return params


class LLMConfigManager:
    """Centralized LLM configuration manager"""

    def __init__(self):
        self._provider = self._get_provider_from_env()
        self._ollama_config = OllamaConfig.from_env()
        self._groq_config = GroqConfig.from_env()
        self._openai_config = OpenAIConfig.from_env()
        self._google_config = GoogleGeminiConfig.from_env()
        self._huggingface_config = HuggingFaceConfig.from_env()
        self._validated = False

    @staticmethod
    def _get_provider_from_env() -> LLMProvider:
        """Get LLM provider from environment variable"""
        provider_str = os.getenv("LLM_PROVIDER", "ollama").lower()

        if provider_str in ["groq_cloud", "groq"]:
            return LLMProvider.GROQ_CLOUD
        elif provider_str == "ollama":
            return LLMProvider.OLLAMA
        elif provider_str == "openai":
            return LLMProvider.OPENAI
        elif provider_str in ["google_gemini", "google", "gemini"]:
            return LLMProvider.GOOGLE_GEMINI
        elif provider_str in ["huggingface", "hf"]:
            return LLMProvider.HUGGINGFACE
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

    @property
    def openai_config(self) -> OpenAIConfig:
        """Get OpenAI configuration"""
        return self._openai_config

    @property
    def google_config(self) -> GoogleGeminiConfig:
        """Get Google Gemini configuration"""
        return self._google_config

    @property
    def huggingface_config(self) -> HuggingFaceConfig:
        """Get Hugging Face configuration"""
        return self._huggingface_config

    def is_ollama_enabled(self) -> bool:
        """Check if Ollama is the current provider"""
        return self._provider == LLMProvider.OLLAMA

    def is_groq_enabled(self) -> bool:
        """Check if Groq Cloud is the current provider"""
        return self._provider == LLMProvider.GROQ_CLOUD

    def is_openai_enabled(self) -> bool:
        """Check if OpenAI is the current provider"""
        return self._provider == LLMProvider.OPENAI

    def is_google_enabled(self) -> bool:
        """Check if Google Gemini is the current provider"""
        return self._provider == LLMProvider.GOOGLE_GEMINI

    def is_huggingface_enabled(self) -> bool:
        """Check if Hugging Face is the current provider"""
        return self._provider == LLMProvider.HUGGINGFACE

    def validate_configuration(self) -> bool:
        """Validate the current configuration"""
        if self._validated:
            return True

        try:
            if self.is_ollama_enabled():
                success = self._validate_ollama()
            elif self.is_groq_enabled():
                success = self._validate_groq()
            elif self.is_openai_enabled():
                success = self._validate_openai()
            elif self.is_google_enabled():
                success = self._validate_google()
            elif self.is_huggingface_enabled():
                success = self._validate_huggingface()
            else:
                logger.error(f"Unknown provider: {self._provider}")
                success = False

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

    def _validate_openai(self) -> bool:
        """Validate OpenAI configuration"""
        if not self.openai_config.api_keys:
            logger.error("No OpenAI API keys configured")
            return False

        # Test API key validity (basic check)
        api_key = self.openai_config.get_current_api_key()
        if not api_key or len(api_key.strip()) < 10:
            logger.error("Invalid OpenAI API key format")
            return False

        logger.info(
            f"OpenAI configuration validated with {len(self.openai_config.api_keys)} API key(s)"
        )
        return True

    def _validate_google(self) -> bool:
        """Validate Google Gemini configuration"""
        if not self.google_config.api_keys:
            logger.error("No Google API keys configured")
            return False

        # Test API key validity (basic check)
        api_key = self.google_config.get_current_api_key()
        if not api_key or len(api_key.strip()) < 10:
            logger.error("Invalid Google API key format")
            return False

        logger.info(
            f"Google Gemini configuration validated with {len(self.google_config.api_keys)} API key(s)"
        )
        return True

    def _validate_huggingface(self) -> bool:
        """Validate Hugging Face configuration"""
        if not self.huggingface_config.model_id:
            logger.error("No Hugging Face model ID configured")
            return False

        # Basic validation of model ID format
        if not "/" in self.huggingface_config.model_id:
            logger.error("Invalid Hugging Face model ID format (should be 'org/model')")
            return False

        logger.info(
            f"Hugging Face configuration validated for model: {self.huggingface_config.model_id}"
        )
        return True

    def get_active_config(
        self,
    ) -> Union[
        OllamaConfig, GroqConfig, OpenAIConfig, GoogleGeminiConfig, HuggingFaceConfig
    ]:
        """Get the configuration for the active provider"""
        if self.is_ollama_enabled():
            return self.ollama_config
        elif self.is_groq_enabled():
            return self.groq_config
        elif self.is_openai_enabled():
            return self.openai_config
        elif self.is_google_enabled():
            return self.google_config
        elif self.is_huggingface_enabled():
            return self.huggingface_config
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    def get_langchain_params(self) -> Dict[str, Any]:
        """Get LangChain parameters for the active provider"""
        if self.is_ollama_enabled():
            return self.ollama_config.to_langchain_params()
        elif self.is_groq_enabled():
            return self.groq_config.to_langchain_params()
        elif self.is_openai_enabled():
            return self.openai_config.to_langchain_params()
        elif self.is_google_enabled():
            return self.google_config.to_langchain_params()
        elif self.is_huggingface_enabled():
            return self.huggingface_config.to_langchain_params()
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

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
        status = {
            "provider": self._provider.value,
            "validated": self._validated,
            "ollama_config": {
                "api_url": self.ollama_config.api_url,
                "primary_model": self.ollama_config.primary_model,
                "backup_model": self.ollama_config.backup_model,
            },
            "groq_config": {
                "primary_model": self.groq_config.primary_model,
                "current_key_index": self.groq_config.current_key_index,
                "keys_count": len(self.groq_config.api_keys),
            },
            "openai_config": {
                "primary_model": self.openai_config.primary_model,
                "current_key_index": self.openai_config.current_key_index,
                "keys_count": len(self.openai_config.api_keys),
            },
            "google_config": {
                "primary_model": self.google_config.primary_model,
                "current_key_index": self.google_config.current_key_index,
                "keys_count": len(self.google_config.api_keys),
            },
            "huggingface_config": {
                "model_id": self.huggingface_config.model_id,
                "task": self.huggingface_config.task,
                "device": self.huggingface_config.device,
            },
        }

        # Add availability checks for non-current providers
        if not self._validated:
            if self._provider != LLMProvider.OLLAMA:
                try:
                    status["ollama_available"] = self._validate_ollama()
                except:
                    status["ollama_available"] = False

        return status


# Global instance
llm_config_manager = LLMConfigManager()


def get_llm_config() -> LLMConfigManager:
    """Get the global LLM configuration manager"""
    return llm_config_manager


def configure_llm_provider(provider: str) -> bool:
    """Configure LLM provider from string"""
    try:
        provider_lower = provider.lower()
        if provider_lower == "ollama":
            return llm_config_manager.switch_provider(LLMProvider.OLLAMA)
        elif provider_lower in ["groq", "groq_cloud"]:
            return llm_config_manager.switch_provider(LLMProvider.GROQ_CLOUD)
        elif provider_lower == "openai":
            return llm_config_manager.switch_provider(LLMProvider.OPENAI)
        elif provider_lower in ["google", "gemini", "google_gemini"]:
            return llm_config_manager.switch_provider(LLMProvider.GOOGLE_GEMINI)
        elif provider_lower in ["huggingface", "hf"]:
            return llm_config_manager.switch_provider(LLMProvider.HUGGINGFACE)
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


def is_openai_enabled() -> bool:
    """Check if OpenAI is enabled"""
    return llm_config_manager.is_openai_enabled()


def is_google_enabled() -> bool:
    """Check if Google Gemini is enabled"""
    return llm_config_manager.is_google_enabled()


def is_huggingface_enabled() -> bool:
    """Check if Hugging Face is enabled"""
    return llm_config_manager.is_huggingface_enabled()


def get_ollama_config() -> OllamaConfig:
    """Get Ollama configuration"""
    return llm_config_manager.ollama_config


def get_groq_config() -> GroqConfig:
    """Get Groq configuration"""
    return llm_config_manager.groq_config


def get_openai_config() -> OpenAIConfig:
    """Get OpenAI configuration"""
    return llm_config_manager.openai_config


def get_google_config() -> GoogleGeminiConfig:
    """Get Google Gemini configuration"""
    return llm_config_manager.google_config


def get_huggingface_config() -> HuggingFaceConfig:
    """Get Hugging Face configuration"""
    return llm_config_manager.huggingface_config
