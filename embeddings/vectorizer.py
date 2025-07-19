# resume_api/core/vectorizer.py
"""
BACKWARD COMPATIBILITY MODULE

This module maintains backward compatibility with existing code that imports
Vectorizer and AddUserDataVectorizer classes. New code should use the centralized
embedding classes from embeddings.manager module.

For new implementations, use:
    from embeddings import EmbeddingManager, ResumeVectorizer, AddUserDataVectorizer as AddUserDataVectorizerNew
"""

import warnings
from typing import List, Dict

# Import the new centralized classes
from .manager import (
    EmbeddingManager,
    ResumeVectorizer,
    AddUserDataVectorizer as AddUserDataVectorizerNew,
)


class Vectorizer:
    """
    BACKWARD COMPATIBILITY CLASS

    This class maintains compatibility with existing code.
    New code should use ResumeVectorizer from embeddings.manager
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Issue a deprecation warning
        warnings.warn(
            "Vectorizer class is deprecated. Use embeddings.ResumeVectorizer instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Use the new centralized approach internally
        self._manager = EmbeddingManager()
        self._vectorizer = ResumeVectorizer(self._manager)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for the given text"""
        return self._vectorizer.generate_embedding(text)

    def generate_total_resume_text(self, resume_data: Dict) -> str:
        """Generate a comprehensive text representation of the entire resume"""
        return self._vectorizer.generate_total_resume_text(resume_data)

    def generate_resume_embeddings(self, resume_data: Dict) -> Dict:
        """Generate embeddings for searchable fields in resume and a combined vector for the entire resume"""
        return self._vectorizer.generate_resume_embeddings(resume_data)

    def generate_total_resume_vector(self, resume_data: Dict) -> List[float]:
        """Generate a combined vector representation of the entire resume"""
        return self._vectorizer.generate_total_resume_vector(resume_data)


class AddUserDataVectorizer:
    """
    BACKWARD COMPATIBILITY CLASS

    This class maintains compatibility with existing code.
    New code should use AddUserDataVectorizer from embeddings.manager
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Issue a deprecation warning
        warnings.warn(
            "AddUserDataVectorizer class is deprecated. Use embeddings.AddUserDataVectorizer from manager instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Use the new centralized approach internally
        self._manager = EmbeddingManager()
        self._vectorizer = AddUserDataVectorizerNew(self._manager)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for the given text"""
        return self._vectorizer.generate_embedding(text)

    def generate_resume_embeddings(self, resume_data: Dict) -> Dict:
        """Generate embeddings for searchable fields in resume"""
        return self._vectorizer.generate_resume_embeddings(resume_data)
