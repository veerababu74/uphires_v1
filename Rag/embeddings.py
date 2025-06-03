from typing import List
from embeddings.vectorizer import Vectorizer


class VectorizerEmbeddingAdapter:
    """Adapter to make our internal Vectorizer compatible with LangChain embeddings interface"""

    def __init__(self, vectorizer: Vectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return [self.vectorizer.generate_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        return self.vectorizer.generate_embedding(text)
