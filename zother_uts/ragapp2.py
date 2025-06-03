# -*- coding: utf-8 -*-
"""Modified RAG Application using LangChain, MongoDB Atlas Vector Search, Internal Vectorizer, and Groq Cloud.

This script sets up a RAG pipeline tailored to specific requirements:
1. Retrieves documents from MongoDB based on vector similarity in the 'combined_resume_vector' field.
2. Extracts a predefined set of fields from the retrieved documents.
3. Passes only these extracted fields as context to the LLM (Groq Cloud).
4. Instructs the LLM to identify the best matching document based on the query and return only its '_id' in JSON format.

Prerequisites:
1.  A MongoDB Atlas cluster with a vector search index named 'resume_vector' configured for the 'combined_resume_vector' field.
2.  Python packages installed: langchain, langchain-mongodb, langchain-groq,
    langchain-community, pymongo, python-dotenv, sentence-transformers
    You can install them using pip:
    pip install langchain langchain-mongodb langchain-groq langchain-community pymongo python-dotenv sentence-transformers
3.  Environment variables set:
    - MONGO_URI: Your MongoDB Atlas connection string.
    - GROQ_API_KEYS: Your Groq API keys (comma-separated).
    - DB_NAME: The name of your MongoDB database.
    - COLLECTION_NAME: The name of your MongoDB collection.
    # Note: VECTOR_FIELD should match your index configuration ('combined_resume_vector').
    - VECTOR_FIELD: The name of the field containing vectors (e.g., 'combined_resume_vector').
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from pymongo import MongoClient
from bson import ObjectId
from pydantic import BaseModel, Field

# Modern LangChain imports
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Core modules from the codebase
from core.custom_logger import CustomLogger
from core.config import AppConfig
from core.helpers import JSONEncoder
from dotenv import load_dotenv
from properties.mango import MONGODB_URI, DB_NAME, COLLECTION_NAME

# Import the internal vectorizer for embeddings
from embeddings.vectorizer import Vectorizer

# --- Custom Embedding Adapter for LangChain ---


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


# --- Configuration ---

load_dotenv()

# Initialize logger
logger_instance = CustomLogger()
logger = logger_instance.get_logger("rag_application")

# Configuration with fallbacks using AppConfig pattern
MONGODB_URI = os.environ.get(
    "MONGO_URI", AppConfig.MONGODB_CONNECTION_STRING or "YOUR_MONGO_URI_HERE"
)
DB_NAME = os.environ.get("DB_NAME", "YOUR_DB_NAME_HERE")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "YOUR_COLLECTION_NAME_HERE")
VECTOR_FIELD = os.environ.get("VECTOR_FIELD", "combined_resume_vector")
INDEX_NAME = "vector_search_index"
GROQ_API_KEY = (
    os.environ.get("GROQ_API_KEYS", "").split(",")[0].strip()
)  # Get first Groq API key

# Fields to extract from retrieved documents and pass to LLM
# IMPORTANT: Ensure these field names exactly match your MongoDB document structure.
FIELDS_TO_EXTRACT = [
    "_id",
    "user_id",
    "username",
    "contact_details",
    "total_experience",
    "notice_period",
    "currency",
    "pay_duration",
    "current_salary",
    "hike",
    "expected_salary",
    "skills",
    "may_also_known_skills",
    "labels",
    "experience",
    "academic_details",
    "source",
    "last_working_day",
    "is_tier1_mba",
    "is_tier1_engineering",
]


# Pydantic models for structured output
class BestMatchResult(BaseModel):
    """Pydantic model for the LLM output"""

    id: str = Field(
        alias="_id", description="MongoDB ObjectId of the best matching candidate"
    )

    class Config:
        populate_by_name = True  # Allow using both 'id' and '_id'


# Basic configuration checks with logging
if not MONGODB_URI or MONGODB_URI == "YOUR_MONGO_URI_HERE":
    logger.warning("MONGO_URI environment variable not set or using placeholder.")
if not DB_NAME or DB_NAME == "YOUR_DB_NAME_HERE":
    logger.warning("DB_NAME environment variable not set or using placeholder.")
if not COLLECTION_NAME or COLLECTION_NAME == "YOUR_COLLECTION_NAME_HERE":
    logger.warning("COLLECTION_NAME environment variable not set or using placeholder.")
if not GROQ_API_KEY:
    logger.error(
        "GROQ_API_KEYS environment variable not set. Please set it to use Groq Cloud models."
    )

# --- Initialize Components ---

logger.info("Initializing RAG components...")


class RAGApplication:
    """Modern RAG Application class with improved error handling and logging"""

    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.retrieval_chain = None
        self.client = None
        self.collection = None
        self.vectorizer = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all RAG components"""
        try:
            self._initialize_embeddings()
            self._initialize_database()
            self._initialize_vector_store()
            self._initialize_llm()
            self._setup_retrieval_chain()
            logger.info("RAG Application initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Application: {e}")
            raise

    def _initialize_embeddings(self):
        """Initialize internal vectorizer for embeddings"""
        try:
            # Initialize the internal vectorizer (same as used throughout the codebase)
            self.vectorizer = Vectorizer()
            # Create LangChain-compatible adapter
            self.embeddings = VectorizerEmbeddingAdapter(self.vectorizer)
            logger.info(
                "Internal vectorizer (SentenceTransformer all-MiniLM-L6-v2) initialized successfully"
            )
        except Exception as e:
            logger.error(f"Error initializing internal vectorizer: {e}")
            logger.info(
                "Make sure sentence-transformers is installed: pip install sentence-transformers"
            )
            raise

    def _initialize_database(self):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(MONGODB_URI)
            db = self.client[DB_NAME]
            self.collection = db[COLLECTION_NAME]  # Test connection
            self.client.admin.command("ping")
            logger.info(
                f"Connected to MongoDB database '{DB_NAME}' and collection '{COLLECTION_NAME}'"
            )
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise

    def _initialize_vector_store(self):
        """Initialize MongoDB Atlas Vector Search"""
        if self.embeddings is None or self.collection is None:
            raise ValueError("Embeddings and collection must be initialized first")

        try:
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name=INDEX_NAME,
                text_key="combined_resume",  # Use the full resume text field
                embedding_key=VECTOR_FIELD,
            )
            logger.info(
                f"MongoDB Atlas Vector Search initialized with index '{INDEX_NAME}' on field '{VECTOR_FIELD}'"
            )
        except Exception as e:
            logger.error(f"Error initializing MongoDBAtlasVectorSearch: {e}")
            raise

    def _initialize_llm(self):
        """Initialize Groq Cloud LLM model"""
        if not GROQ_API_KEY:
            logger.warning("Skipping LLM initialization due to missing Groq API key")
            return

        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model="gemma2-9b-it",  # Using same model as internal GroqcloudLLM
                temperature=0.0,  # Low temp for precise extraction
            )
            logger.info("Groq Cloud LLM (gemma2-9b-it) initialized")
        except Exception as e:
            logger.error(f"Error initializing Groq LLM: {e}")
            raise

    def _setup_retrieval_chain(self):
        """Setup the modern LangChain retrieval chain with LCEL"""
        if not self.llm:
            logger.warning("LLM not available, skipping chain setup")
            return
        # Define the prompt template with structured output
        prompt_template_text = """You are an assistant analyzing candidate resume data to find the best match for a query.
Based on the user's question and the following candidate data snippets, identify the single best matching candidate.
Return ONLY the MongoDB '_id' of the best matching candidate in JSON format.

{format_instructions}

Candidate Data Snippets:
---
{context}
---

User Question: {question}

Best Matching Candidate:"""

        # Setup output parser
        output_parser = JsonOutputParser(pydantic_object=BestMatchResult)

        # Create prompt template
        prompt = PromptTemplate(
            template=prompt_template_text,
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )

        # Create the chain using LCEL (LangChain Expression Language)
        self.retrieval_chain = prompt | self.llm | output_parser
        logger.info("Retrieval chain setup completed")

    def _normalize_field_value(self, value) -> str:
        """Normalize field values for consistent processing"""
        if value is None:
            return ""
        if isinstance(value, (list, dict)):
            return json.dumps(value, cls=JSONEncoder)
        return str(value).strip()

    def _normalize_list_field(self, value) -> List[str]:
        """Normalize list fields for consistent processing"""
        if not value:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [self._normalize_field_value(item) for item in value if item]
        return [str(value)]

    def get_relevant_ids_and_context(
        self, question: str, k: int = 3
    ) -> Tuple[List[ObjectId], Optional[str]]:
        """Retrieves relevant documents, fetches specified fields, and formats them as context."""
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot retrieve documents.")
            return [], None

        logger.info(f"Retrieving top {k} documents for question: {question}")
        try:
            # Retrieve documents using similarity search
            retrieved_docs = self.vector_store.similarity_search(query=question, k=k)

            if not retrieved_docs:
                logger.warning("No relevant documents found.")
                return [], None

            # Extract _ids from metadata
            doc_ids = []
            for doc in retrieved_docs:
                if hasattr(doc, "metadata") and "_id" in doc.metadata:
                    doc_id = doc.metadata["_id"]
                    if not isinstance(doc_id, ObjectId):
                        try:
                            doc_id = ObjectId(doc_id)
                        except Exception:
                            logger.warning(
                                f"Could not convert metadata _id '{doc_id}' to ObjectId. Skipping."
                            )
                            continue
                    doc_ids.append(doc_id)
                else:
                    logger.warning(f"Retrieved document missing _id in metadata: {doc}")

            if not doc_ids:
                logger.error("Could not extract valid _ids from retrieved documents.")
                return [], None

            logger.info(f"Retrieved document _ids: {doc_ids}")

            # Fetch the full documents with specified fields from MongoDB
            projection = {field: 1 for field in FIELDS_TO_EXTRACT}
            if "_id" not in projection:
                projection["_id"] = 1

            fetched_docs_cursor = self.collection.find(
                {"_id": {"$in": doc_ids}}, projection
            )  # Format the context string with normalization
            context_parts = []
            logger.info("=== DOCUMENTS FETCHED FROM MONGODB ===")
            for i, doc in enumerate(fetched_docs_cursor):
                logger.info(f"Document {i+1} raw data:")
                logger.info(json.dumps(doc, indent=2, cls=JSONEncoder))

                # Normalize document fields
                normalized_doc = {}
                for field, value in doc.items():
                    if field == "_id":
                        normalized_doc[field] = str(value)
                    elif field in ["skills", "may_also_known_skills", "labels"]:
                        normalized_doc[field] = self._normalize_list_field(value)
                    else:
                        normalized_doc[field] = self._normalize_field_value(value)

                logger.info(f"Document {i+1} normalized data:")
                logger.info(json.dumps(normalized_doc, indent=2, cls=JSONEncoder))

                context_parts.append(
                    json.dumps(normalized_doc, indent=2, cls=JSONEncoder)
                )
            logger.info("=== END OF DOCUMENTS ===")

            context_string = "\n\n---\n\n".join(context_parts)
            logger.info("Formatted context for LLM")

            # Print the context retrieved from MongoDB
            logger.info("=== CONTEXT RETRIEVED FROM MONGODB ===")
            logger.info(f"Number of documents retrieved: {len(context_parts)}")
            logger.info("Context content:")
            logger.info(context_string)
            logger.info("=== END OF CONTEXT ===")

            return doc_ids, context_string

        except Exception as e:
            logger.error(f"Error during retrieval or context preparation: {e}")
            return [], None

    def ask_resume_question_and_get_id(self, question: str) -> Optional[Dict]:
        """Processes a question, retrieves context, asks LLM, and returns the ID in JSON format."""
        if not self.retrieval_chain:
            logger.error(
                "Retrieval chain is not available. Cannot process the question."
            )
            return None

        logger.info(f"Processing question: {question}")
        retrieved_ids, context = self.get_relevant_ids_and_context(question)

        if context is None or not retrieved_ids:
            logger.error("Failed to get context or retrieve IDs.")
            return None

        try:
            logger.info("Invoking retrieval chain...")
            # Use the modern LCEL chain
            result = self.retrieval_chain.invoke(
                {"context": context, "question": question}
            )
            # The JsonOutputParser should return a dict directly
            if isinstance(result, dict) and ("_id" in result or "id" in result):
                # Normalize the result to always use "_id"
                if "id" in result and "_id" not in result:
                    result["_id"] = result.pop("id")
                logger.info("Successfully processed question")
                logger.info(f"Final Answer (JSON): {json.dumps(result, indent=2)}")
                return result
            else:
                logger.warning(f"Unexpected result format: {result}")
                return {"error": "Unexpected result format", "raw_result": str(result)}

        except Exception as e:
            logger.error(f"Error during LLM processing: {e}")
            # Fallback: Try to find any retrieved ID in the error message
            for doc_id in retrieved_ids:
                if str(doc_id) in str(e):
                    logger.info(
                        f"Found retrieved ID {doc_id} in error message. Returning as fallback."
                    )
                    return {"_id": str(doc_id)}
            return {"error": str(e)}

    def health_check(self) -> Dict[str, bool]:
        """Check the health of all components"""
        return {
            "embeddings_initialized": self.embeddings is not None,
            "vectorizer_initialized": self.vectorizer is not None,
            "vector_store_initialized": self.vector_store is not None,
            "llm_initialized": self.llm is not None,
            "retrieval_chain_initialized": self.retrieval_chain is not None,
            "mongodb_connection": self.client is not None,
        }


# --- Main Execution Example ---

# Global instance for backward compatibility
rag_app = None


def initialize_rag_application() -> RAGApplication:
    """Initialize the RAG application"""
    global rag_app
    if rag_app is None:
        rag_app = RAGApplication()
    return rag_app


def ask_resume_question_and_get_id(question: str) -> Optional[Dict]:
    """Backward compatibility function"""
    global rag_app
    if rag_app is None:
        rag_app = initialize_rag_application()
    return rag_app.ask_resume_question_and_get_id(question)


def get_relevant_ids_and_context(
    question: str, k: int = 3
) -> Tuple[List[ObjectId], Optional[str]]:
    """Backward compatibility function"""
    global rag_app
    if rag_app is None:
        rag_app = initialize_rag_application()
    return rag_app.get_relevant_ids_and_context(question, k)


if __name__ == "__main__":
    try:  # Check configuration
        if (
            MONGODB_URI != "YOUR_MONGO_URI_HERE"
            and GROQ_API_KEY
            and DB_NAME != "YOUR_DB_NAME_HERE"
            and COLLECTION_NAME != "YOUR_COLLECTION_NAME_HERE"
        ):
            # Initialize RAG application
            logger.info("--- Initializing RAG Application ---")
            rag_application = initialize_rag_application()

            # Health check
            health_status = rag_application.health_check()
            logger.info(f"Health Check: {health_status}")

            # Test query
            sample_question = "Find the candidate with 5 yeas above experince"
            logger.info("--- Running Sample Query ---")
            result = rag_application.ask_resume_question_and_get_id(sample_question)

            if result:
                logger.info("--- Query Finished --- Final Result ---")
                logger.info(f"Result: {json.dumps(result, indent=2)}")
            else:
                logger.warning("--- Query Finished --- No result obtained.")
        else:
            logger.error("Configuration Incomplete:")
            logger.error(
                "Please ensure MONGO_URI, GROQ_API_KEYS, DB_NAME, COLLECTION_NAME are set."
            )
            logger.info("Run the script (e.g., python rag.py) to test.")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

logger.info("RAG Application setup complete. Ready for execution or function calls.")
