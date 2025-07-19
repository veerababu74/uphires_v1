# embeddings/config.py
"""
Configuration management for embedding providers
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers"""

    provider: str = "sentence_transformer"
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    device: str = "cpu"
    api_key: Optional[str] = None
    trust_remote_code: bool = False

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create configuration from environment variables"""
        provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformer").lower()

        # Default configurations for different providers
        if provider == "sentence_transformer":
            model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
            embedding_dim = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
            device = os.getenv("EMBEDDING_DEVICE", "cpu")
            trust_remote_code = (
                os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
            )

            return cls(
                provider=provider,
                model_name=model_name,
                embedding_dimension=embedding_dim,
                device=device,
                trust_remote_code=trust_remote_code,
            )

        elif provider == "openai":
            model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            embedding_dim = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
            api_key = os.getenv("OPENAI_API_KEY")

            return cls(
                provider=provider,
                model_name=model_name,
                embedding_dimension=embedding_dim,
                api_key=api_key,
            )

        else:
            # Default to sentence transformer
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "api_key": self.api_key,
            "trust_remote_code": self.trust_remote_code,
        }

    def validate(self) -> bool:
        """Validate the configuration"""
        if self.provider in ["openai", "jina", "google_genai"]:
            if not self.api_key:
                raise ValueError(f"API key is required for {self.provider} provider")

        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

        return True


# Predefined configurations for common models
EMBEDDING_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "provider": "sentence_transformer",
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "device": "cpu",
    },
    "qwen-embedding-0.6b": {
        "provider": "sentence_transformer",
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dimension": 1024,
        "device": "cpu",
    },
    "baai-bge-large-zh": {
        "provider": "sentence_transformer",
        "model_name": "BAAI/bge-large-zh-v1.5",
        "embedding_dimension": 1024,
        "device": "cpu",
    },
    # RECOMMENDED MODELS FOR PRODUCTION (Fast + High Accuracy + 1024 dims)
    "BAAI/bge-large-en-v1.5": {
        "dimensions": 1024,
        "trust_remote_code": True,
        "description": "🥇 BEST OVERALL - Top MTEB performance, fast inference, 1024 dims",
    },
    "thenlper/gte-large": {
        "dimensions": 1024,
        "trust_remote_code": False,
        "description": "🥈 EXCELLENT SPEED - Very fast, great accuracy, multilingual support",
    },
    "BAAI/bge-large-zh-v1.5": {
        "dimensions": 1024,
        "trust_remote_code": True,
        "description": "🇨🇳 CHINESE OPTIMIZED - Best for Chinese text, your original target model",
    },
    "intfloat/e5-large-v2": {
        "dimensions": 1024,
        "trust_remote_code": False,
        "description": "⚡ FAST & RELIABLE - Excellent speed/accuracy balance, stable performance",
    },
    "nomic-ai/nomic-embed-text-v1": {
        "provider": "sentence_transformer",
        "model_name": "nomic-ai/nomic-embed-text-v1",
        "embedding_dimension": 768,
        "device": "cpu",
        "trust_remote_code": True,
    },
    "e5-small-v2": {
        "provider": "sentence_transformer",
        "model_name": "intfloat/e5-small-v2",
        "embedding_dimension": 384,
        "device": "cpu",
    },
    "e5-base-v2": {
        "provider": "sentence_transformer",
        "model_name": "intfloat/e5-base-v2",
        "embedding_dimension": 768,
        "device": "cpu",
    },
    "e5-large-v2": {
        "provider": "sentence_transformer",
        "model_name": "intfloat/e5-large-v2",
        "embedding_dimension": 1024,
        "device": "cpu",
    },
    "all-mpnet-base-v2": {
        "provider": "sentence_transformer",
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "embedding_dimension": 768,
        "device": "cpu",
    },
    "all-roberta-large-v1": {
        "provider": "sentence_transformer",
        "model_name": "sentence-transformers/all-roberta-large-v1",
        "embedding_dimension": 1024,
        "device": "cpu",
    },
    # BEST PERFORMING MODELS FOR ACCURACY
    "bge-large-en-v1.5": {
        "provider": "sentence_transformer",
        "model_name": "BAAI/bge-large-en-v1.5",
        "embedding_dimension": 1024,
        "device": "cpu",
        "description": "Best English model by BAAI - Top performance on MTEB benchmark",
    },
    "gte-large": {
        "provider": "sentence_transformer",
        "model_name": "thenlper/gte-large",
        "embedding_dimension": 1024,
        "device": "cpu",
        "description": "Alibaba's GTE-large - Excellent for retrieval tasks",
    },
    "bge-base-en-v1.5": {
        "provider": "sentence_transformer",
        "model_name": "BAAI/bge-base-en-v1.5",
        "embedding_dimension": 768,
        "device": "cpu",
        "description": "BAAI base model - Good balance of speed and accuracy",
    },
    "multilingual-e5-large": {
        "provider": "sentence_transformer",
        "model_name": "intfloat/multilingual-e5-large",
        "embedding_dimension": 1024,
        "device": "cpu",
        "description": "Best multilingual model - Supports 100+ languages",
    },
    "bge-m3": {
        "provider": "sentence_transformer",
        "model_name": "BAAI/bge-m3",
        "embedding_dimension": 1024,
        "device": "cpu",
        "description": "BAAI M3 - Multi-lingual, multi-functionality, multi-granularity",
    },
    "gte-base": {
        "provider": "sentence_transformer",
        "model_name": "thenlper/gte-base",
        "embedding_dimension": 768,
        "device": "cpu",
        "description": "Alibaba's GTE-base - Fast and accurate",
    },
    "e5-mistral-7b-instruct": {
        "provider": "sentence_transformer",
        "model_name": "intfloat/e5-mistral-7b-instruct",
        "embedding_dimension": 4096,
        "device": "cpu",
        "description": "Large 7B parameter model - Highest accuracy but slower",
    },
    "openai-small": {
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "embedding_dimension": 1536,
    },
}


def get_config_by_name(config_name: str) -> EmbeddingConfig:
    """Get predefined configuration by name"""
    if config_name not in EMBEDDING_CONFIGS:
        raise ValueError(
            f"Unknown configuration: {config_name}. Available: {list(EMBEDDING_CONFIGS.keys())}"
        )

    config_dict = EMBEDDING_CONFIGS[config_name]
    return EmbeddingConfig(**config_dict)


def list_available_configs() -> Dict[str, Dict[str, Any]]:
    """List all available predefined configurations"""
    return EMBEDDING_CONFIGS.copy()
