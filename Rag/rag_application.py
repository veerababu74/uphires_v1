import json
import os
from typing import List, Dict, Optional, Tuple
from bson import ObjectId
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaLLM
from core.custom_logger import CustomLogger
from core.helpers import JSONEncoder
from core.llm_factory import create_llm  # Use centralized LLM factory
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

    # def _initialize_llm(self):
    #     """Initialize Groq Cloud LLM model"""
    #     if not RAGConfig.GROQ_API_KEY:
    #         logger.warning("Skipping LLM initialization due to missing Groq API key")
    #         return

    #     try:
    #         self.llm = ChatGroq(
    #             api_key=RAGConfig.GROQ_API_KEY,
    #             model=RAGConfig.LLM_MODEL,
    #             temperature=RAGConfig.LLM_TEMPERATURE,
    #         )
    #         logger.info(f"Groq Cloud LLM ({RAGConfig.LLM_MODEL}) initialized")
    #     except Exception as e:
    #         logger.error(f"Error initializing Groq LLM: {e}")
    #         raise
    def _initialize_llm(self):
        """Initialize LLM using centralized factory"""
        try:
            # Use the centralized LLM factory which handles provider selection
            self.llm = create_llm()

            from core.llm_config import get_llm_config

            config_manager = get_llm_config()
            provider_name = config_manager.provider.value

            logger.info(f"LLM initialized using {provider_name} provider")

        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            logger.info("Attempting fallback LLM initialization...")

            # Fallback to legacy Ollama initialization with model fallback logic
            try:
                self.llm = self._initialize_ollama_with_fallback()
            except Exception as fallback_error:
                logger.error(
                    f"Fallback LLM initialization also failed: {fallback_error}"
                )
                raise Exception(
                    f"Failed to initialize any LLM: {e}, fallback: {fallback_error}"
                )

    def _initialize_ollama_with_fallback(self):
        """Initialize Ollama LLM with model fallback logic similar to other parts of the system"""
        import requests

        # Define model fallback order similar to other components
        primary_model = RAGConfig.OLLAMA_MODEL
        backup_model = RAGConfig.OLLAMA_BACKUP_MODEL
        fallback_model = RAGConfig.OLLAMA_FALLBACK_MODEL

        # Check Ollama connection first
        try:
            response = requests.get(f"{RAGConfig.OLLAMA_API_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama service not accessible")

            # Get available models
            models_data = response.json()
            available_models = [
                model["name"] for model in models_data.get("models", [])
            ]
            logger.info(f"Available Ollama models: {available_models}")

        except Exception as e:
            logger.error(f"Failed to connect to Ollama service: {e}")
            raise Exception("Ollama service not accessible")

        # Try primary model first
        selected_model = primary_model
        if primary_model not in available_models:
            logger.warning(f"{primary_model} not found.")
            logger.info(f"Available models: {available_models}")

            # Try backup model
            if backup_model in available_models:
                logger.info(f"Using backup model: {backup_model}")
                selected_model = backup_model
            # Try fallback model
            elif fallback_model in available_models:
                logger.info(f"Using fallback model: {fallback_model}")
                selected_model = fallback_model
            else:
                # Look for any available qwen model as last resort
                qwen_models = [
                    model for model in available_models if "qwen" in model.lower()
                ]
                if qwen_models:
                    selected_model = qwen_models[0]
                    logger.info(f"Using available qwen model: {selected_model}")
                else:
                    error_msg = (
                        f"None of the configured models ({primary_model}, {backup_model}, {fallback_model}) "
                        f"are available. Available models: {available_models}. "
                        f"Please pull a compatible model using: ollama pull {primary_model}"
                    )
                    logger.error(error_msg)
                    raise Exception(error_msg)
        else:
            logger.info(f"Using primary model: {primary_model}")

        # Initialize OllamaLLM with the selected model
        try:
            llm = OllamaLLM(
                model=selected_model,
                temperature=RAGConfig.OLLAMA_TEMPERATURE,
                base_url=RAGConfig.OLLAMA_API_URL,
                timeout=RAGConfig.OLLAMA_TIMEOUT,
                request_timeout=RAGConfig.OLLAMA_REQUEST_TIMEOUT,
            )
            logger.info(
                f"Ollama LLM initialized successfully with model: {selected_model}"
            )
            return llm

        except Exception as e:
            logger.error(
                f"Failed to initialize Ollama LLM with model {selected_model}: {e}"
            )
            raise

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
