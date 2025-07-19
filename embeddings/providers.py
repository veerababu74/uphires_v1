# embeddings/providers.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import os
import logging

from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """SentenceTransformer-based embedding provider"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self._model = None

        # Set up local model cache directory
        self.cache_dir = self._get_model_cache_dir()
        self._ensure_cache_dir_exists()
        self._model = None

        # Update embedding dimension based on model
        if "Qwen3-Embedding-0.6B" in model_name:
            self.embedding_dim = 1024
        elif "bge-large-zh-v1.5" in model_name:
            self.embedding_dim = 1024
        elif "bge-large-en-v1.5" in model_name:
            self.embedding_dim = 1024
        elif "bge-base-en-v1.5" in model_name:
            self.embedding_dim = 768
        elif "bge-m3" in model_name:
            self.embedding_dim = 1024
        elif "gte-large" in model_name:
            self.embedding_dim = 1024
        elif "gte-base" in model_name:
            self.embedding_dim = 768
        elif "multilingual-e5-large" in model_name:
            self.embedding_dim = 1024
        elif "e5-mistral-7b-instruct" in model_name:
            self.embedding_dim = 4096
        elif "nomic-embed-text-v1" in model_name:
            self.embedding_dim = 768  # nomic-ai/nomic-embed-text-v1
        elif "e5-small-v2" in model_name:
            self.embedding_dim = 384  # intfloat/e5-small-v2
        elif "e5-base-v2" in model_name:
            self.embedding_dim = 768  # intfloat/e5-base-v2
        elif "e5-large-v2" in model_name:
            self.embedding_dim = 1024  # intfloat/e5-large-v2
        elif "all-mpnet-base-v2" in model_name:
            self.embedding_dim = 768  # sentence-transformers/all-mpnet-base-v2
        elif "all-roberta-large-v1" in model_name:
            self.embedding_dim = 1024  # sentence-transformers/all-roberta-large-v1
        elif "text-embedding-3-small" in model_name:
            self.embedding_dim = 1536
        elif "jina-embeddings-v3" in model_name:
            self.embedding_dim = 1024
        elif "text-embedding-004" in model_name:
            self.embedding_dim = 768

    def _get_model_cache_dir(self) -> str:
        """Get the local cache directory for models"""
        # Create emmodels directory in project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "emmodels")

        # Create model-specific subdirectory (sanitize model name for filesystem)
        model_dir_name = self.model_name.replace("/", "_").replace(":", "_")
        model_cache_dir = os.path.join(models_dir, model_dir_name)

        return model_cache_dir

    def _ensure_cache_dir_exists(self):
        """Ensure the cache directory exists"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Model cache directory: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Could not create cache directory {self.cache_dir}: {e}")
            # Fall back to default behavior if cache dir creation fails
            self.cache_dir = None

    def _is_model_cached(self) -> bool:
        """Check if model is already cached locally"""
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return False

        # Check for essential SentenceTransformer files
        essential_files = [
            "config_sentence_transformers.json",  # SentenceTransformer config
            "modules.json",  # SentenceTransformer modules
        ]

        # Check for model files (at least one should exist)
        model_files = ["pytorch_model.bin", "model.safetensors"]

        # Check for tokenizer files
        tokenizer_files = ["tokenizer.json", "vocab.txt"]

        # Must have essential files
        has_essential = all(
            os.path.exists(os.path.join(self.cache_dir, file))
            for file in essential_files
        )

        # Must have at least one model file
        has_model = any(
            os.path.exists(os.path.join(self.cache_dir, file)) for file in model_files
        )

        # Must have at least one tokenizer file
        has_tokenizer = any(
            os.path.exists(os.path.join(self.cache_dir, file))
            for file in tokenizer_files
        )

        return has_essential and has_model and has_tokenizer

    @property
    def model(self):
        """Lazy loading of the model with caching support"""
        if self._model is None:
            try:
                # Check if model is cached locally
                if self._is_model_cached():
                    logger.info(f"Loading cached model from: {self.cache_dir}")
                    # Load from local cache directory
                    self._model = SentenceTransformer(
                        self.cache_dir,
                        device=self.device,
                        trust_remote_code=self.trust_remote_code,
                    )
                else:
                    # Download model and save to cache directory
                    logger.info(
                        f"Downloading model {self.model_name} to cache: {self.cache_dir}"
                    )

                    # First download to default location
                    temp_model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        trust_remote_code=self.trust_remote_code,
                    )

                    # Save the model to our cache directory
                    temp_model.save(self.cache_dir)
                    logger.info(f"Model saved to cache: {self.cache_dir}")

                    # Now load from our cache
                    self._model = SentenceTransformer(
                        self.cache_dir,
                        device=self.device,
                        trust_remote_code=self.trust_remote_code,
                    )

                    # Verify caching was successful
                    if self._is_model_cached():
                        logger.info(f"Model {self.model_name} successfully cached")
                    else:
                        logger.warning(
                            f"Model caching verification failed for {self.model_name}"
                        )

                logger.info(
                    f"SentenceTransformer model loaded successfully: {self.model_name}"
                )
            except Exception as e:
                logger.error(
                    f"Error loading SentenceTransformer model {self.model_name}: {e}"
                )
                # Fall back to default loading without cache
                logger.info("Falling back to default model loading...")
                try:
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        trust_remote_code=self.trust_remote_code,
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback loading also failed: {fallback_error}")
                    raise
        return self._model

    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for the given text"""
        if not text or text == "N/A" or text.strip() == "":
            return np.zeros(self.embedding_dim).tolist()

        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim).tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings generated by this provider"""
        return self.embedding_dim

    def get_provider_name(self) -> str:
        """Get the name of the embedding provider"""
        return f"SentenceTransformer ({self.model_name})"


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI-based embedding provider"""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_dim = 1536  # Default for text-embedding-3-small

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for the given text using OpenAI"""
        if not text or text == "N/A" or text.strip() == "":
            return np.zeros(self.embedding_dim).tolist()

        try:
            # This would require openai library
            # from openai import OpenAI
            # client = OpenAI(api_key=self.api_key)
            # response = client.embeddings.create(input=text, model=self.model_name)
            # return response.data[0].embedding

            # For now, fallback to zero vector
            logger.warning("OpenAI embedding not implemented, returning zero vector")
            return np.zeros(self.embedding_dim).tolist()
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            return np.zeros(self.embedding_dim).tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings generated by this provider"""
        return self.embedding_dim

    def get_provider_name(self) -> str:
        """Get the name of the embedding provider"""
        return f"OpenAI ({self.model_name})"


class EmbeddingProviderFactory:
    """Factory for creating embedding providers"""

    @staticmethod
    def create_provider(
        provider_type: str = "sentence_transformer", **kwargs
    ) -> BaseEmbeddingProvider:
        """Create an embedding provider based on type"""
        provider_type = provider_type.lower()

        if provider_type == "sentence_transformer":
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            device = kwargs.get("device", "cpu")
            trust_remote_code = kwargs.get("trust_remote_code", False)
            return SentenceTransformerProvider(
                model_name=model_name,
                device=device,
                trust_remote_code=trust_remote_code,
            )

        elif provider_type == "openai":
            model_name = kwargs.get("model_name", "text-embedding-3-small")
            api_key = kwargs.get("api_key")
            return OpenAIEmbeddingProvider(model_name=model_name, api_key=api_key)

        else:
            raise ValueError(f"Unsupported embedding provider: {provider_type}")

    @staticmethod
    def create_default_provider() -> BaseEmbeddingProvider:
        """Create default embedding provider based on environment"""
        from .config import EmbeddingConfig

        config = EmbeddingConfig.from_env()
        config.validate()

        if config.provider == "sentence_transformer":
            return SentenceTransformerProvider(
                model_name=config.model_name,
                device=config.device,
                trust_remote_code=getattr(config, "trust_remote_code", False),
            )

        elif config.provider == "openai":
            return OpenAIEmbeddingProvider(
                model_name=config.model_name, api_key=config.api_key
            )

        else:
            logger.warning(
                f"Unknown embedding provider {config.provider}, falling back to SentenceTransformer"
            )
            return SentenceTransformerProvider()
