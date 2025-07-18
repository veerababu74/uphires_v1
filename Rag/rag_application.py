import json
import os
from typing import List, Dict, Optional, Tuple
from bson import ObjectId
from core.custom_logger import CustomLogger
from core.helpers import JSONEncoder
from core.llm_factory import LLMFactory  # Use centralized LLM factory
from core.llm_config import LLMConfigManager, LLMProvider
from core.exceptions import LLMProviderError
from embeddings.vectorizer import Vectorizer

from .config import RAGConfig
from .models import SearchStatistics
from .embeddings import VectorizerEmbeddingAdapter
from .database import DatabaseManager
from .chains import ChainManager
from .utils import DocumentProcessor
from .search_engines import VectorSearchEngine, LLMSearchEngine

logger = CustomLogger().get_logger("rag_application")


class RAGApplication:
    """Modern RAG Application with improved architecture"""

    def __init__(self):
        self.embeddings = None
        self.database_manager = None
        self.chain_manager = None
        self.llm = None
        self.vector_search_engine = None
        self.llm_search_engine = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all RAG components"""
        try:
            self._initialize_embeddings()
            self._initialize_database()
            self._initialize_llm()
            self._initialize_chains()
            self._initialize_search_engines()
            logger.info("RAG Application initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Application: {e}")
            raise

    def _initialize_embeddings(self):
        """Initialize internal vectorizer for embeddings"""
        try:
            vectorizer = Vectorizer()
            self.embeddings = VectorizerEmbeddingAdapter(vectorizer)
            logger.info(
                "Internal vectorizer (SentenceTransformer all-MiniLM-L6-v2) initialized successfully"
            )
        except Exception as e:
            logger.error(f"Error initializing internal vectorizer: {e}")
            raise

    def _initialize_database(self):
        """Initialize database manager"""
        self.database_manager = DatabaseManager(self.embeddings)

    def _initialize_llm(self):
        """Initialize LLM using centralized factory"""
        try:
            # Initialize LLM configuration manager
            self.llm_manager = LLMConfigManager()

            # Use the centralized LLM factory which handles provider selection
            self.llm = LLMFactory.create_llm()

            provider_name = self.llm_manager.provider.value
            logger.info(f"RAG LLM initialized using {provider_name} provider")

        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise LLMProviderError(f"Failed to initialize RAG LLM: {e}")

    def _initialize_chains(self):
        """Initialize chain manager"""
        if not self.llm:
            logger.warning("LLM not available, skipping chain setup")
            return

        self.chain_manager = ChainManager(self.llm)
        logger.info("Chain manager initialized successfully")

    def _initialize_search_engines(self):
        """Initialize search engines"""
        self.vector_search_engine = VectorSearchEngine(
            self.database_manager.vector_store, self.database_manager.collection
        )

        if self.chain_manager:
            self.llm_search_engine = LLMSearchEngine(
                self.database_manager.vector_store,
                self.database_manager.collection,
                self.chain_manager,
            )

    # Public API methods
    def vector_similarity_search(self, query: str, limit: int = 50) -> Dict:
        """Perform pure vector similarity search"""
        if not self.vector_search_engine:
            return {"error": "Vector search engine not initialized"}

        return self.vector_search_engine.search(query, limit)

    def llm_context_search(self, query: str, context_size: int = 5) -> Dict:
        """Perform LLM-based search with user-controlled context size"""
        if not self.llm_search_engine:
            return {"error": "LLM search engine not initialized"}

        return self.llm_search_engine.search(query, context_size)

    def ask_resume_question_with_limits(
        self,
        question: str,
        mongodb_retrieval_limit: int = RAGConfig.DEFAULT_MONGODB_LIMIT,
        llm_context_limit: int = RAGConfig.DEFAULT_LLM_LIMIT,
    ) -> Optional[Dict]:
        """Enhanced version of ask_resume_question with separate MongoDB and LLM limits"""
        if not self.llm_search_engine:
            logger.error("LLM search engine not available")
            return None

        return self.llm_search_engine.ask_question_with_limits(
            question, mongodb_retrieval_limit, llm_context_limit
        )

    def get_candidates_with_limits(
        self,
        question: str,
        mongodb_retrieval_limit: int = RAGConfig.DEFAULT_MONGODB_LIMIT,
        llm_context_limit: int = RAGConfig.DEFAULT_LLM_LIMIT,
        max_results: int = RAGConfig.DEFAULT_MAX_RESULTS,
    ) -> Optional[Dict]:
        """Get candidate IDs with separate control for MongoDB retrieval and LLM processing"""
        if not self.llm_search_engine:
            logger.error("LLM search engine not available")
            return None

        return self.llm_search_engine.get_candidates_with_limits(
            question, mongodb_retrieval_limit, llm_context_limit, max_results
        )

    def rank_all_candidates_with_limits(
        self,
        question: str,
        mongodb_retrieval_limit: int = 100,
        llm_context_limit: int = 50,
    ) -> Optional[Dict]:
        """Rank candidates with separate control for MongoDB retrieval and LLM processing"""
        if not self.llm_search_engine:
            logger.error("LLM search engine not available")
            return None

        return self.llm_search_engine.rank_candidates_with_limits(
            question, mongodb_retrieval_limit, llm_context_limit
        )

    def switch_llm_provider(self, new_provider: str) -> bool:
        """Switch to a different LLM provider and reinitialize components.

        Args:
            new_provider (str): New provider to use ('groq', 'ollama', 'openai', 'google', 'huggingface')

        Returns:
            bool: True if switch was successful, False otherwise
        """
        try:
            # Update the LLM configuration manager
            provider_map = {
                "ollama": LLMProvider.OLLAMA,
                "groq": LLMProvider.GROQ_CLOUD,
                "groq_cloud": LLMProvider.GROQ_CLOUD,
                "openai": LLMProvider.OPENAI,
                "google": LLMProvider.GOOGLE_GEMINI,
                "gemini": LLMProvider.GOOGLE_GEMINI,
                "google_gemini": LLMProvider.GOOGLE_GEMINI,
                "huggingface": LLMProvider.HUGGINGFACE,
                "hf": LLMProvider.HUGGINGFACE,
            }

            if new_provider.lower() not in provider_map:
                logger.error(f"Unsupported LLM provider: {new_provider}")
                return False

            new_provider_enum = provider_map[new_provider.lower()]

            logger.info(
                f"Switching RAG LLM provider from {self.llm_manager.provider.value} to {new_provider_enum.value}"
            )

            # Update provider in configuration
            self.llm_manager.provider = new_provider_enum

            # Reinitialize LLM
            self._initialize_llm()

            # Reinitialize chain manager with new LLM
            self._initialize_chains()

            # Reinitialize search engines
            self._initialize_search_engines()

            logger.info(
                f"Successfully switched RAG to {new_provider_enum.value} provider"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to switch RAG LLM provider: {e}")
            return False
