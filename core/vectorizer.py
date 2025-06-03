from embeddings.vectorizer import Vectorizer
from typing import Optional

# Global vectorizer instance
_vectorizer: Optional[Vectorizer] = None


def get_vectorizer() -> Vectorizer:
    """Get or create the vectorizer instance"""
    global _vectorizer

    if _vectorizer is None:
        _vectorizer = Vectorizer()

    return _vectorizer
