# embeddings/__init__.py
"""
Centralized embedding module for UPHires application.

This module provides a centralized way to handle vector embeddings across the application.
It supports multiple embedding providers (SentenceTransformer, OpenAI, etc.) and provides
specialized vectorizers for different resume data formats.

Main Components:
- EmbeddingManager: Central manager for all embedding operations
- ResumeVectorizer: For standard resume data format
- AddUserDataVectorizer: For user-added resume data format
- Vectorizer, AddUserDataVectorizer: Backward compatibility classes

Usage:
    from embeddings import EmbeddingManager, ResumeVectorizer

    # Use the new centralized approach
    manager = EmbeddingManager()
    vectorizer = ResumeVectorizer(manager)

    # Or use backward compatibility
    from embeddings import Vectorizer
    vectorizer = Vectorizer()
"""

# New centralized classes
from .manager import (
    EmbeddingManager,
    ResumeVectorizer,
    AddUserDataVectorizer as AddUserDataVectorizerNew,
)
from .providers import EmbeddingProviderFactory, SentenceTransformerProvider
from .base import BaseEmbeddingProvider, BaseVectorizer
from .config import EmbeddingConfig, get_config_by_name, list_available_configs

# Backward compatibility - import original classes
from .vectorizer import Vectorizer, AddUserDataVectorizer

# Make the most commonly used classes available at module level
__all__ = [
    # New centralized classes
    "EmbeddingManager",
    "ResumeVectorizer",
    "AddUserDataVectorizerNew",
    "EmbeddingProviderFactory",
    "SentenceTransformerProvider",
    "BaseEmbeddingProvider",
    "BaseVectorizer",
    "EmbeddingConfig",
    "get_config_by_name",
    "list_available_configs",
    # Backward compatibility
    "Vectorizer",
    "AddUserDataVectorizer",
]

# Default instances for easy access
_default_manager = None
_default_resume_vectorizer = None
_default_add_user_data_vectorizer = None


def get_default_embedding_manager() -> EmbeddingManager:
    """Get the default embedding manager instance (singleton)"""
    global _default_manager
    if _default_manager is None:
        _default_manager = EmbeddingManager()
    return _default_manager


def get_default_resume_vectorizer() -> ResumeVectorizer:
    """Get the default resume vectorizer instance (singleton)"""
    global _default_resume_vectorizer
    if _default_resume_vectorizer is None:
        _default_resume_vectorizer = ResumeVectorizer(get_default_embedding_manager())
    return _default_resume_vectorizer


def get_default_add_user_data_vectorizer() -> AddUserDataVectorizerNew:
    """Get the default add user data vectorizer instance (singleton)"""
    global _default_add_user_data_vectorizer
    if _default_add_user_data_vectorizer is None:
        _default_add_user_data_vectorizer = AddUserDataVectorizerNew(
            get_default_embedding_manager()
        )
    return _default_add_user_data_vectorizer
