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
import time
import asyncio
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


class CandidateMatch(BaseModel):
    """Pydantic model for a single candidate match with score"""

    id: str = Field(alias="_id", description="MongoDB ObjectId of the candidate")
    relevance_score: float = Field(
        description="Relevance score from 0.0 to 1.0, where 1.0 is perfect match"
    )
    match_reason: str = Field(
        description="Brief explanation of why this candidate matches the query"
    )

    class Config:
        populate_by_name = True


class AllMatchesResult(BaseModel):
    """Pydantic model for all matching candidates ranked by relevance"""

    total_candidates: int = Field(description="Total number of candidates analyzed")
    matches: List[CandidateMatch] = Field(
        description="List of all matching candidates ranked by relevance score (highest first)"
    )


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
        self.ranking_chain = None  # New chain for ranking all matches
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
        # Define the prompt template with structured output for best match
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

        # Setup output parser for best match
        output_parser = JsonOutputParser(pydantic_object=BestMatchResult)

        # Create prompt template for best match
        prompt = PromptTemplate(
            template=prompt_template_text,
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )  # Create the chain using LCEL (LangChain Expression Language)
        self.retrieval_chain = prompt | self.llm | output_parser

        # Setup ranking chain for all matches
        ranking_prompt_template = """You are an expert AI recruiter analyzing candidate resumes to rank ALL candidates based on their relevance to a specific search query.

CRITICAL INSTRUCTIONS:
1. You MUST analyze EVERY single candidate document provided in the context (separated by ---DOCUMENT_SEPARATOR---)
2. Score each candidate from 0.0 to 1.0 based on how well they match the query:
   - 1.0 = Perfect match (has all required skills/experience exactly)
   - 0.8-0.9 = Excellent match (has most required skills/experience with good alignment)
   - 0.6-0.7 = Good match (has some required skills/experience with decent alignment)
   - 0.4-0.5 = Partial match (has related skills/experience with some relevance)
   - 0.1-0.3 = Weak match (minimal relevance, few matching criteria)
   - 0.0 = No match (no relevant skills or experience)
3. Provide a clear, specific reason for each score explaining WHY this candidate matches or doesn't match
4. Return ALL candidates ranked by relevance score (highest first)
5. Do not skip any candidates - analyze every single one provided
6. Count the documents carefully and ensure you analyze all of them

Consider these factors when scoring (in order of importance):
- Technical skills match (programming languages, frameworks, tools, technologies)
- Experience level and years of experience matching requirements
- Industry/domain experience relevance
- Educational background relevance (degrees, certifications)
- Specific project/work experience alignment
- Notice period and availability
- Salary expectations alignment
- Geographic location/work preferences

IMPORTANT: Each document is separated by '---DOCUMENT_SEPARATOR---'. Make sure to process EVERY document.

{format_instructions}

Search Query: {question}

ALL Candidate Documents to Analyze:
---
{context}
---

Now analyze and rank EVERY SINGLE candidate above:"""

        # Setup output parser for all matches
        ranking_output_parser = JsonOutputParser(pydantic_object=AllMatchesResult)

        # Create prompt template for ranking all matches
        ranking_prompt = PromptTemplate(
            template=ranking_prompt_template,
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": ranking_output_parser.get_format_instructions()
            },
        )

        # Create the ranking chain
        self.ranking_chain = ranking_prompt | self.llm | ranking_output_parser

        logger.info("Retrieval chain and ranking chain setup completed")

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

    def get_relevant_ids_and_context_enhanced(
        self,
        question: str,
        mongodb_retrieval_limit: int = 50,
        llm_context_limit: int = 10,
    ) -> Tuple[List[ObjectId], Optional[str], Dict[str, int]]:
        """
        Enhanced version with separate control for MongoDB retrieval and LLM context.

        Args:
            question: The search query
            mongodb_retrieval_limit: Maximum documents to retrieve from MongoDB (default: 50)
            llm_context_limit: Maximum documents to send to LLM (default: 10)

        Returns:
            Tuple of (document_ids, context_string, statistics)
            where statistics contains counts of documents at each stage
        """
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot retrieve documents.")
            return [], None, {"mongodb_retrieved": 0, "llm_context_sent": 0}

        logger.info(
            f"Retrieving up to {mongodb_retrieval_limit} documents from MongoDB for question: {question}"
        )
        logger.info(
            f"Will send up to {llm_context_limit} documents to LLM for processing"
        )

        try:
            # Step 1: Retrieve documents from MongoDB using vector search
            retrieved_docs = self.vector_store.similarity_search(
                query=question, k=mongodb_retrieval_limit
            )

            if not retrieved_docs:
                logger.warning("No relevant documents found in MongoDB.")
                return [], None, {"mongodb_retrieved": 0, "llm_context_sent": 0}

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
                return [], None, {"mongodb_retrieved": 0, "llm_context_sent": 0}

            mongodb_count = len(doc_ids)
            logger.info(
                f"Successfully retrieved {mongodb_count} documents from MongoDB"
            )

            # Step 2: Limit documents for LLM context
            llm_doc_ids = doc_ids[:llm_context_limit]
            llm_count = len(llm_doc_ids)

            logger.info(f"Limiting to {llm_count} documents for LLM context")
            if llm_count < mongodb_count:
                logger.info(
                    f"Note: {mongodb_count - llm_count} documents were retrieved from MongoDB but not sent to LLM"
                )

            # Step 3: Fetch the selected documents with specified fields from MongoDB
            projection = {field: 1 for field in FIELDS_TO_EXTRACT}
            if "_id" not in projection:
                projection["_id"] = 1

            fetched_docs_cursor = self.collection.find(
                {"_id": {"$in": llm_doc_ids}}, projection
            )

            # Format the context string with normalization
            context_parts = []
            logger.info("=== DOCUMENTS BEING SENT TO LLM ===")
            for i, doc in enumerate(fetched_docs_cursor):
                logger.info(f"LLM Document {i+1}/{llm_count} raw data:")
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

                logger.info(f"LLM Document {i+1}/{llm_count} normalized data:")
                logger.info(json.dumps(normalized_doc, indent=2, cls=JSONEncoder))

                context_parts.append(
                    json.dumps(normalized_doc, indent=2, cls=JSONEncoder)
                )
            logger.info("=== END OF LLM DOCUMENTS ===")

            context_string = "\n\n---\n\n".join(context_parts)
            logger.info("Formatted context for LLM")

            # Print the context statistics
            logger.info("=== CONTEXT STATISTICS ===")
            logger.info(f"Documents retrieved from MongoDB: {mongodb_count}")
            logger.info(f"Documents sent to LLM: {llm_count}")
            logger.info(f"Context length: {len(context_string)} characters")
            logger.info("=== END OF STATISTICS ===")

            statistics = {
                "mongodb_retrieved": mongodb_count,
                "llm_context_sent": llm_count,
                "context_length": len(context_string),
            }

            return doc_ids, context_string, statistics

        except Exception as e:
            logger.error(f"Error during enhanced retrieval or context preparation: {e}")
            return [], None, {"mongodb_retrieved": 0, "llm_context_sent": 0}

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

    def ask_resume_question_with_limits(
        self,
        question: str,
        mongodb_retrieval_limit: int = 50,
        llm_context_limit: int = 10,
    ) -> Optional[Dict]:
        """
        Enhanced version of ask_resume_question_and_get_id with separate MongoDB and LLM limits.

        Args:
            question: The search query
            mongodb_retrieval_limit: Maximum documents to retrieve from MongoDB (default: 50)
            llm_context_limit: Maximum documents to send to LLM (default: 10)

        Returns:
            Dict containing the best matching document ID and processing statistics
        """
        if not self.retrieval_chain:
            logger.error(
                "Retrieval chain is not available. Cannot process the question."
            )
            return None

        logger.info(
            f"Processing question with limits - MongoDB: {mongodb_retrieval_limit}, LLM: {llm_context_limit}"
        )
        logger.info(f"Question: {question}")

        retrieved_ids, context, statistics = self.get_relevant_ids_and_context_enhanced(
            question, mongodb_retrieval_limit, llm_context_limit
        )

        if context is None or not retrieved_ids:
            logger.error("Failed to get context or retrieve IDs.")
            return {
                "error": "No documents found or context preparation failed",
                "statistics": statistics,
            }

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

                # Add processing statistics
                result["statistics"] = statistics
                result["processing_info"] = {
                    "mongodb_retrieval_limit": mongodb_retrieval_limit,
                    "llm_context_limit": llm_context_limit,
                    "documents_considered": statistics["mongodb_retrieved"],
                    "documents_analyzed_by_llm": statistics["llm_context_sent"],
                }

                logger.info("Successfully processed question with enhanced limits")
                logger.info(f"Final Answer (JSON): {json.dumps(result, indent=2)}")
                return result
            else:
                logger.warning(f"Unexpected result format: {result}")
                return {
                    "error": "Unexpected result format",
                    "raw_result": str(result),
                    "statistics": statistics,
                }

        except Exception as e:
            logger.error(f"Error during LLM processing: {e}")
            # Fallback: Try to find any retrieved ID in the error message
            for doc_id in retrieved_ids[
                :llm_context_limit
            ]:  # Only consider LLM context docs
                if str(doc_id) in str(e):
                    logger.info(
                        f"Found retrieved ID {doc_id} in error message. Returning as fallback."
                    )
                    return {
                        "_id": str(doc_id),
                        "statistics": statistics,
                        "fallback": True,
                    }
            return {"error": str(e), "statistics": statistics}

    def get_candidates_with_limits(
        self,
        question: str,
        mongodb_retrieval_limit: int = 50,
        llm_context_limit: int = 10,
        max_results: int = 20,
    ) -> Optional[Dict]:
        """
        Get candidate IDs with separate control for MongoDB retrieval and LLM processing.

        Args:
            question: The search query
            mongodb_retrieval_limit: Maximum documents to retrieve from MongoDB (default: 50)
            llm_context_limit: Maximum documents to send to LLM (default: 10)
            max_results: Maximum number of results to return (default: 20)

        Returns:
            Dict containing candidate IDs and processing statistics
        """
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot retrieve documents.")
            return None

        logger.info(
            f"Retrieving candidates with limits - MongoDB: {mongodb_retrieval_limit}, LLM: {llm_context_limit}"
        )

        try:
            # Get relevant documents and their IDs with enhanced limits
            retrieved_ids, context, statistics = (
                self.get_relevant_ids_and_context_enhanced(
                    question, mongodb_retrieval_limit, llm_context_limit
                )
            )

            if not retrieved_ids:
                logger.warning("No relevant documents found.")
                return {"message": "No candidates found", "statistics": statistics}

            # Create the result dictionary with all retrieved IDs
            result = {
                "total_mongodb_retrieved": statistics["mongodb_retrieved"],
                "total_llm_processed": statistics["llm_context_sent"],
                "processing_info": {
                    "mongodb_retrieval_limit": mongodb_retrieval_limit,
                    "llm_context_limit": llm_context_limit,
                },
            }

            # Add all retrieved IDs (from MongoDB)
            for i, doc_id in enumerate(retrieved_ids):
                result[f"id{i+1}"] = str(doc_id)

            logger.info(
                f"Successfully retrieved {len(retrieved_ids)} candidate IDs from MongoDB"
            )
            logger.info(f"LLM would process {statistics['llm_context_sent']} documents")
            logger.info(f"Result: {json.dumps(result, indent=2)}")

            return result

        except Exception as e:
            logger.error(f"Error during candidate ID retrieval with limits: {e}")
            return {"error": str(e)}

    def rank_all_candidates_with_limits(
        self,
        question: str,
        mongodb_retrieval_limit: int = 100,
        llm_context_limit: int = 50,
    ) -> Optional[Dict]:
        """
        Rank candidates with separate control for MongoDB retrieval and LLM processing.

        Args:
            question: The search query
            mongodb_retrieval_limit: Maximum documents to retrieve from MongoDB (default: 100)
            llm_context_limit: Maximum documents to send to LLM for processing (default: 50)

        Returns:
            Dict containing all ranked candidates with processing statistics
        """
        if not self.ranking_chain:
            logger.error("Ranking chain is not available. Cannot process the question.")
            return None

        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot retrieve documents.")
            return None

        logger.info(
            f"Ranking candidates with limits - MongoDB: {mongodb_retrieval_limit}, LLM: {llm_context_limit}"
        )
        logger.info(f"Query: '{question}'")

        try:
            # Get documents with enhanced limits
            retrieved_ids, context, statistics = (
                self.get_relevant_ids_and_context_enhanced(
                    question, mongodb_retrieval_limit, llm_context_limit
                )
            )

            if not retrieved_ids or context is None:
                logger.warning("No relevant documents found in MongoDB.")
                return {
                    "total_candidates": 0,
                    "query": question,
                    "matches": [],
                    "message": "No candidates found matching the query",
                    "statistics": statistics,
                }

            logger.info(
                f"Retrieved {statistics['mongodb_retrieved']} candidates from MongoDB"
            )
            logger.info(
                f"Sending {statistics['llm_context_sent']} candidates to LLM for comprehensive ranking..."
            )

            # Use the ranking chain to analyze the selected candidates
            result = self.ranking_chain.invoke(
                {"context": context, "question": question}
            )

            # Validate and format the result
            if isinstance(result, dict):
                if "matches" in result and "total_candidates" in result:
                    # Ensure all matches have string IDs and are properly formatted
                    formatted_matches = []
                    for match in result.get("matches", []):
                        if isinstance(match, dict) and "_id" in match:
                            formatted_match = {
                                "_id": str(match["_id"]),
                                "relevance_score": float(
                                    match.get("relevance_score", 0.0)
                                ),
                                "match_reason": str(
                                    match.get("match_reason", "No reason provided")
                                ),
                            }
                            formatted_matches.append(formatted_match)

                    # Add processing statistics and limits info
                    final_result = {
                        "total_candidates": result.get(
                            "total_candidates", len(formatted_matches)
                        ),
                        "query": question,
                        "matches": formatted_matches,
                        "statistics": statistics,
                        "processing_info": {
                            "mongodb_retrieval_limit": mongodb_retrieval_limit,
                            "llm_context_limit": llm_context_limit,
                            "mongodb_retrieved": statistics["mongodb_retrieved"],
                            "llm_analyzed": statistics["llm_context_sent"],
                        },
                    }

                    logger.info(
                        f"Successfully ranked {len(formatted_matches)} candidates"
                    )
                    return final_result

                else:
                    logger.error(f"Invalid ranking result format: {result}")
                    return {
                        "error": "Invalid ranking result format",
                        "raw_result": str(result),
                        "statistics": statistics,
                    }
            else:
                logger.error(f"Unexpected ranking result type: {type(result)}")
                return {
                    "error": "Unexpected ranking result type",
                    "raw_result": str(result),
                    "statistics": statistics,
                }

        except Exception as e:
            logger.error(f"Error during candidate ranking with limits: {e}")
            return {
                "error": str(e),
                "statistics": (
                    statistics
                    if "statistics" in locals()
                    else {"mongodb_retrieved": 0, "llm_context_sent": 0}
                ),
            }

    def retrieve_documents_only(
        self, question: str, mongodb_retrieval_limit: int = 50
    ) -> List[Dict]:
        """
        Performs vector search retrieval and returns retrieved documents with scores.
        This function does NOT involve the LLM.

        Args:
            question: The search query
            mongodb_retrieval_limit: Maximum documents to retrieve from MongoDB (default: 50)

        Returns:
            List of dictionaries, each containing _id, score, and text_key for retrieved documents,
            sorted by score in descending order.
        """
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot retrieve documents.")
            return []

        logger.info(
            f"Performing retrieval-only search for question: '{question}' (limit: {mongodb_retrieval_limit})"
        )

        try:
            # Retrieve documents using similarity search. LangChain's similarity_search usually returns
            # a list of Document objects with page_content and metadata, including the score.
            retrieved_docs_with_score = self.vector_store.similarity_search_with_score(
                query=question, k=mongodb_retrieval_limit
            )

            if not retrieved_docs_with_score:
                logger.warning("No relevant documents found during retrieval.")
                return []

            results = []
            for doc, score in retrieved_docs_with_score:
                # Extract _id and the text_key field from the Document object
                doc_id = None
                text_content = doc.page_content  # This should be the text_key content

                if hasattr(doc, "metadata") and "_id" in doc.metadata:
                    doc_id = doc.metadata["_id"]
                    if not isinstance(doc_id, ObjectId):
                        try:
                            doc_id = ObjectId(doc_id)
                        except Exception:
                            logger.warning(
                                f"Could not convert metadata _id '{doc_id}' to ObjectId. Skipping."
                            )
                            doc_id = str(doc_id)  # Keep as string if conversion fails
                    else:
                        doc_id = str(
                            doc_id
                        )  # Convert ObjectId to string for easier handling
                else:
                    logger.warning(f"Retrieved document missing _id in metadata: {doc}")
                    # If no _id, use the text content or skip? For now, include what we have.
                    # Depending on exact requirement, we might need to adjust this.
                    pass  # Decide if we must have an _id or can proceed without. Assuming _id is crucial.
                    continue  # Skip if _id is missing/invalid

                results.append(
                    {
                        "_id": doc_id,
                        "score": score,
                        "text_preview": (
                            text_content[:200] + "..."
                            if text_content and len(text_content) > 200
                            else text_content
                        ),  # Include a preview
                        # Potentially include other relevant metadata if needed for display
                    }
                )

            # Sort results by score in descending order
            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

            logger.info(f"Retrieved and sorted {len(sorted_results)} documents.")
            return sorted_results

        except Exception as e:
            logger.error(f"Error during retrieval-only search: {e}")
            return []

    def analyze_specific_documents_with_llm(
        self,
        question: str,
        document_ids: List[str],
    ) -> Optional[Dict]:
        """
        Takes a list of document IDs, fetches their content, and uses the LLM
        to find the best matching ID among ONLY these documents.

        Args:
            question: The search query.
            document_ids: A list of string representation of MongoDB ObjectIds to analyze.

        Returns:
            Dict containing the best matching document ID as determined by the LLM,
            or None if the process fails.
        """
        if not self.retrieval_chain:
            logger.error("Retrieval chain is not available. Cannot analyze documents.")
            return None

        if not document_ids:
            logger.warning("No document IDs provided for analysis.")
            return {"message": "No document IDs provided.", "_id": None}

        logger.info(
            f"Analyzing {len(document_ids)} specific documents with LLM for question: '{question}'"
        )
        logger.info(f"Document IDs to analyze: {document_ids}")

        try:
            # Convert string IDs to ObjectId for MongoDB query
            object_ids = [ObjectId(doc_id) for doc_id in document_ids]

            # Fetch the full documents with specified fields from MongoDB for the given IDs
            projection = {field: 1 for field in FIELDS_TO_EXTRACT}
            if "_id" not in projection:
                projection["_id"] = 1

            fetched_docs_cursor = self.collection.find(
                {"_id": {"$in": object_ids}}, projection
            )

            # Format the context string with normalization
            context_parts = []
            found_ids = []  # Keep track of IDs actually found in DB
            logger.info("=== DOCUMENTS FETCHED FOR LLM ANALYSIS ===")
            for i, doc in enumerate(fetched_docs_cursor):
                found_ids.append(str(doc["_id"]))
                logger.info(
                    f"LLM Analysis Document {i+1}/{len(document_ids)} raw data:"
                )
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

                logger.info(
                    f"LLM Analysis Document {i+1}/{len(document_ids)} normalized data:"
                )
                logger.info(json.dumps(normalized_doc, indent=2, cls=JSONEncoder))

                context_parts.append(
                    json.dumps(normalized_doc, indent=2, cls=JSONEncoder)
                )
            logger.info("=== END OF LLM ANALYSIS DOCUMENTS ===")

            if not context_parts:
                logger.warning(
                    "None of the provided document IDs were found in the database."
                )
                return {
                    "message": "None of the provided document IDs were found.",
                    "_id": None,
                }

            # Log which IDs were provided vs which were found
            missing_ids = list(set(document_ids) - set(found_ids))
            if missing_ids:
                logger.warning(
                    f"The following provided IDs were not found in the database: {missing_ids}"
                )

            context_string = "\n\n---\n\n".join(context_parts)
            logger.info("Formatted context for LLM analysis")

            logger.info("Invoking retrieval chain for specific document analysis...")
            # Use the modern LCEL chain to find the best match among the provided documents
            result = self.retrieval_chain.invoke(
                {"context": context_string, "question": question}
            )

            # The JsonOutputParser should return a dict directly
            if isinstance(result, dict) and ("_id" in result or "id" in result):
                # Normalize the result to always use "_id"
                if "id" in result and "_id" not in result:
                    result["_id"] = result.pop("id")
                logger.info("Successfully analyzed specific documents")
                logger.info(f"Best Match ID (JSON): {json.dumps(result, indent=2)}")
                return result
            else:
                logger.warning(f"Unexpected result format from LLM analysis: {result}")
                return {
                    "error": "Unexpected result format",
                    "raw_result": str(result),
                    "_id": None,
                }

        except Exception as e:
            logger.error(f"Error during specific document analysis: {e}")
            # In case of error, if the LLM output is garbled but contains a valid ID from the input list,
            # we could try to extract it as a fallback, but for now, just return the error.
            return {"error": str(e), "_id": None}


# Global wrapper functions for enhanced RAG methods
# Global instance for backward compatibility
rag_app = None


def ask_resume_question_enhanced(question, mongodb_limit=50, llm_limit=10):
    """
    Enhanced global function to ask resume questions with configurable limits.

    Args:
        question (str): The question to ask about resumes
        mongodb_limit (int): Maximum documents to retrieve from MongoDB (default: 50)
        llm_limit (int): Maximum documents to send to LLM for processing (default: 10)

    Returns:
        dict: Result with answer and processing statistics
    """
    global rag_app
    if rag_app is None:
        rag_app = RAGApplication()

    return rag_app.ask_resume_question_with_limits(
        question=question,
        mongodb_retrieval_limit=mongodb_limit,
        llm_context_limit=llm_limit,
    )


def search_candidates_enhanced(query, mongodb_limit=50, llm_limit=10, max_results=20):
    """
    Enhanced global function to search for candidates with configurable limits.

    Args:
        query (str): Search query for candidates
        mongodb_limit (int): Maximum documents to retrieve from MongoDB (default: 50)        llm_limit (int): Maximum documents to send to LLM for processing (default: 10)
        max_results (int): Maximum number of results to return (default: 20)

    Returns:
        dict: Search results with candidates and processing statistics
    """
    global rag_app
    if rag_app is None:
        rag_app = RAGApplication()

    return rag_app.get_candidates_with_limits(
        question=query,
        mongodb_retrieval_limit=mongodb_limit,
        llm_context_limit=llm_limit,
        max_results=max_results,
    )


def rank_candidates_enhanced(query, mongodb_limit=100, llm_limit=20):
    """
    Enhanced global function to rank all candidates with configurable limits.

    Args:
        query (str): Ranking query
        mongodb_limit (int): Maximum documents to retrieve from MongoDB (default: 100)
        llm_limit (int): Maximum documents to send to LLM for processing (default: 20)    Returns:
        dict: Ranked candidates with processing statistics
    """
    global rag_app
    if rag_app is None:
        rag_app = RAGApplication()

    return rag_app.rank_all_candidates_with_limits(
        question=query,
        mongodb_retrieval_limit=mongodb_limit,
        llm_context_limit=llm_limit,
    )


def get_retrieval_statistics():
    """
    Get statistics about the last retrieval operation.

    Returns:
        dict: Statistics including mongodb_retrieved, llm_context_sent, context_length
    """
    global rag_app
    if rag_app is None:
        return {"error": "RAG application not initialized"}

    # Return last operation statistics if available
    return getattr(rag_app, "_last_stats", {})


# Usage examples and configuration guide
def print_configuration_guide():
    """
    Print a comprehensive guide on how to use the enhanced RAG functions.
    """
    guide = """
    🔧 Enhanced RAG Configuration Guide
    ================================
    
    The enhanced RAG functions allow you to control:
    1. MongoDB Retrieval Limit: How many documents to fetch from the database
    2. LLM Context Limit: How many documents to send to the language model
    
    📊 Performance Optimization Tips:
    
    💾 MongoDB Retrieval Limit:
    - Higher values (50-200): Better recall, more comprehensive search
    - Lower values (10-30): Faster retrieval, reduced database load
    - Default: 50 documents
    
    🧠 LLM Context Limit:
    - Higher values (20-50): More context for better answers, higher cost
    - Lower values (5-15): Faster processing, lower API costs
    - Default: 10 documents
    
    🚀 Usage Examples:
    
    # Basic usage with defaults (MongoDB: 50, LLM: 10)
    result = ask_resume_question_enhanced("Find Python developers")
    
    # High recall search (MongoDB: 100, LLM: 20)
    result = ask_resume_question_enhanced(
        "Find senior engineers", 
        mongodb_limit=100, 
        llm_limit=20
    )
    
    # Fast search (MongoDB: 20, LLM: 5)
    result = search_candidates_enhanced(
        "project manager", 
        mongodb_limit=20, 
        llm_limit=5
    )
    
    # Comprehensive ranking (MongoDB: 200, LLM: 30)
    result = rank_candidates_enhanced(
        "leadership skills", 
        mongodb_limit=200, 
        llm_limit=30
    )
    
    📈 Monitoring Performance:
    stats = get_retrieval_statistics()
    print(f"Retrieved: {stats.get('mongodb_retrieved', 0)} docs")
    print(f"Processed: {stats.get('llm_context_sent', 0)} docs")
    print(f"Context length: {stats.get('context_length', 0)} chars")
    """
    print(guide)


# Configuration presets for different use cases
PERFORMANCE_PRESETS = {
    "fast": {"mongodb_limit": 20, "llm_limit": 5},
    "balanced": {"mongodb_limit": 50, "llm_limit": 10},
    "comprehensive": {"mongodb_limit": 100, "llm_limit": 20},
    "exhaustive": {"mongodb_limit": 200, "llm_limit": 30},
}


def ask_resume_question_preset(question, preset="balanced"):
    """
    Ask resume questions using predefined performance presets.

    Args:
        question (str): The question to ask
        preset (str): Performance preset ("fast", "balanced", "comprehensive", "exhaustive")

    Returns:
        dict: Result with answer and processing statistics
    """
    if preset not in PERFORMANCE_PRESETS:
        raise ValueError(
            f"Invalid preset. Choose from: {list(PERFORMANCE_PRESETS.keys())}"
        )

    config = PERFORMANCE_PRESETS[preset]
    return ask_resume_question_enhanced(
        question, mongodb_limit=config["mongodb_limit"], llm_limit=config["llm_limit"]
    )


def search_candidates_preset(query, preset="balanced", max_results=20):
    """
    Search candidates using predefined performance presets.

    Args:
        query (str): Search query
        preset (str): Performance preset ("fast", "balanced", "comprehensive", "exhaustive")
        max_results (int): Maximum results to return

    Returns:
        dict: Search results with candidates and statistics
    """
    if preset not in PERFORMANCE_PRESETS:
        raise ValueError(
            f"Invalid preset. Choose from: {list(PERFORMANCE_PRESETS.keys())}"
        )

    config = PERFORMANCE_PRESETS[preset]
    return search_candidates_enhanced(
        query,
        mongodb_limit=config["mongodb_limit"],
        llm_limit=config["llm_limit"],
        max_results=max_results,
    )


if __name__ == "__main__":
    # Example usage of enhanced functions
    async def demo_enhanced_features():
        """Demonstrate the enhanced RAG features with configurable limits."""
        print("🔧 Enhanced RAG Features Demo")
        print("=" * 50)

        # Print configuration guide
        print_configuration_guide()

        try:
            # Initialize RAG application
            rag_app = RAGApplication()
            print("✅ RAG Application initialized successfully")

            # Test different configurations
            test_query = "Find Python developers with machine learning experience"

            print(f"\n🧪 Testing query: '{test_query}'")
            print("-" * 50)

            # Test different presets
            presets_to_test = ["fast", "balanced", "comprehensive"]

            for preset in presets_to_test:
                print(f"\n📊 Testing '{preset}' preset:")
                config = PERFORMANCE_PRESETS[preset]
                print(f"   MongoDB limit: {config['mongodb_limit']}")
                print(f"   LLM limit: {config['llm_limit']}")

                try:
                    start_time = time.time()
                    result = search_candidates_preset(
                        test_query, preset=preset, max_results=5
                    )
                    end_time = time.time()

                    if result and "candidates" in result:
                        stats = result.get("processing_info", {})
                        print(f"   ✅ Found {len(result['candidates'])} candidates")
                        print(
                            f"   📈 MongoDB retrieved: {stats.get('mongodb_retrieved', 0)}"
                        )
                        print(
                            f"   🧠 LLM processed: {stats.get('llm_context_sent', 0)}"
                        )
                        print(f"   ⏱️ Time taken: {end_time - start_time:.2f} seconds")
                    else:
                        print(f"   ❌ No results found")

                except Exception as e:
                    print(f"   ❌ Error: {str(e)}")

            # Test custom configuration
            print(f"\n🎛️ Testing custom configuration:")
            print(f"   MongoDB limit: 75, LLM limit: 15")

            try:
                start_time = time.time()
                result = ask_resume_question_enhanced(
                    "What are the key skills mentioned in resumes?",
                    mongodb_limit=75,
                    llm_limit=15,
                )
                end_time = time.time()

                if result and "answer" in result:
                    stats = result.get("processing_info", {})
                    print(f"   ✅ Answer generated successfully")
                    print(
                        f"   📈 MongoDB retrieved: {stats.get('mongodb_retrieved', 0)}"
                    )
                    print(f"   🧠 LLM processed: {stats.get('llm_context_sent', 0)}")
                    print(f"   ⏱️ Time taken: {end_time - start_time:.2f} seconds")
                    print(f"   📝 Answer preview: {result['answer'][:100]}...")
                else:
                    print(f"   ❌ No answer generated")

            except Exception as e:
                print(f"   ❌ Error: {str(e)}")

        except Exception as e:
            print(f"💥 Failed to initialize RAG application: {str(e)}")  # Run the demo

    print("Starting Enhanced RAG Demo...")
    asyncio.run(demo_enhanced_features())


class DocumentRetriever:
    """Class responsible for retrieving documents from MongoDB using vector search."""

    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        """Initialize the document retriever with MongoDB connection details."""
        self.client = None
        self.collection = None
        self.vector_store = None
        self.embeddings = None
        self.vectorizer = None

        # Initialize components
        self._initialize_components(mongo_uri, db_name, collection_name)

    def _initialize_components(
        self, mongo_uri: str, db_name: str, collection_name: str
    ):
        """Initialize MongoDB connection and vector store."""
        try:
            # Initialize vectorizer
            self.vectorizer = Vectorizer()
            self.embeddings = VectorizerEmbeddingAdapter(self.vectorizer)

            # Initialize MongoDB connection
            self.client = MongoClient(mongo_uri)
            db = self.client[db_name]
            self.collection = db[collection_name]
            self.client.admin.command("ping")

            # Initialize vector store
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name="vector_search_index",
                text_key="combined_resume",
                embedding_key="combined_resume_vector",
            )

            logger.info(
                f"DocumentRetriever initialized successfully for {db_name}.{collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize DocumentRetriever: {e}")
            raise

    def retrieve_documents(self, query: str, limit: int = 50) -> Dict:
        """
        Retrieve documents based on vector similarity search.

        Args:
            query: The search query
            limit: Maximum number of documents to retrieve

        Returns:
            Dict containing:
            - documents: List of retrieved documents with scores
            - total_count: Total number of documents retrieved
            - query: Original query
            - retrieval_info: Additional retrieval statistics
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return {
                "error": "Vector store not initialized",
                "documents": [],
                "total_count": 0,
                "query": query,
            }

        try:
            # Perform vector similarity search
            retrieved_docs_with_score = self.vector_store.similarity_search_with_score(
                query=query, k=limit
            )

            if not retrieved_docs_with_score:
                logger.warning("No documents found for query")
                return {
                    "documents": [],
                    "total_count": 0,
                    "query": query,
                    "retrieval_info": {"limit": limit, "search_time": 0},
                }

            # Process and format results
            results = []
            for doc, score in retrieved_docs_with_score:
                if hasattr(doc, "metadata") and "_id" in doc.metadata:
                    doc_id = doc.metadata["_id"]
                    if not isinstance(doc_id, ObjectId):
                        try:
                            doc_id = ObjectId(doc_id)
                        except Exception:
                            logger.warning(
                                f"Could not convert _id '{doc_id}' to ObjectId"
                            )
                            doc_id = str(doc_id)
                    else:
                        doc_id = str(doc_id)

                    results.append(
                        {
                            "_id": doc_id,
                            "score": score,
                            "text_preview": (
                                doc.page_content[:200] + "..."
                                if doc.page_content and len(doc.page_content) > 200
                                else doc.page_content
                            ),
                        }
                    )

            # Sort by score in descending order
            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

            return {
                "documents": sorted_results,
                "total_count": len(sorted_results),
                "query": query,
                "retrieval_info": {
                    "limit": limit,
                    "search_time": time.time(),  # Add actual timing if needed
                },
            }

        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            return {"error": str(e), "documents": [], "total_count": 0, "query": query}

    def get_document_details(self, document_ids: List[str]) -> Dict:
        """
        Get full details for specific document IDs.

        Args:
            document_ids: List of document IDs to fetch

        Returns:
            Dict containing document details and metadata
        """
        if not self.collection:
            logger.error("MongoDB collection not initialized")
            return {"error": "Collection not initialized", "documents": []}

        try:
            # Convert string IDs to ObjectId
            object_ids = [ObjectId(doc_id) for doc_id in document_ids]

            # Fetch documents with specified fields
            projection = {field: 1 for field in FIELDS_TO_EXTRACT}
            if "_id" not in projection:
                projection["_id"] = 1

            documents = list(
                self.collection.find({"_id": {"$in": object_ids}}, projection)
            )

            # Convert ObjectIds to strings for JSON serialization
            for doc in documents:
                doc["_id"] = str(doc["_id"])

            return {
                "documents": documents,
                "total_count": len(documents),
                "requested_ids": document_ids,
                "found_ids": [str(doc["_id"]) for doc in documents],
            }

        except Exception as e:
            logger.error(f"Error fetching document details: {e}")
            return {"error": str(e), "documents": []}


class LLMAnalyzer:
    """Class responsible for analyzing documents using LLM."""

    def __init__(self, groq_api_key: str):
        """Initialize the LLM analyzer with Groq API key."""
        self.llm = None
        self.retrieval_chain = None
        self.ranking_chain = None

        # Initialize components
        self._initialize_components(groq_api_key)

    def _initialize_components(self, groq_api_key: str):
        """Initialize LLM and analysis chains."""
        try:
            # Initialize Groq LLM
            self.llm = ChatGroq(
                api_key=groq_api_key, model="gemma2-9b-it", temperature=0.0
            )

            # Setup retrieval chain
            self._setup_retrieval_chain()

            # Setup ranking chain
            self._setup_ranking_chain()

            logger.info("LLMAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLMAnalyzer: {e}")
            raise

    def _setup_retrieval_chain(self):
        """Setup the retrieval chain for finding best matching document."""
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

        output_parser = JsonOutputParser(pydantic_object=BestMatchResult)
        prompt = PromptTemplate(
            template=prompt_template_text,
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )

        self.retrieval_chain = prompt | self.llm | output_parser

    def _setup_ranking_chain(self):
        """Setup the ranking chain for analyzing multiple documents."""
        ranking_prompt_template = """You are an expert AI recruiter analyzing candidate resumes to rank ALL candidates based on their relevance to a specific search query.

CRITICAL INSTRUCTIONS:
1. You MUST analyze EVERY single candidate document provided in the context (separated by ---DOCUMENT_SEPARATOR---)
2. Score each candidate from 0.0 to 1.0 based on how well they match the query
3. Provide a clear, specific reason for each score
4. Return ALL candidates ranked by relevance score (highest first)
5. Do not skip any candidates - analyze every single one provided

{format_instructions}

Search Query: {question}

ALL Candidate Documents to Analyze:
---
{context}
---

Now analyze and rank EVERY SINGLE candidate above:"""

        ranking_output_parser = JsonOutputParser(pydantic_object=AllMatchesResult)
        ranking_prompt = PromptTemplate(
            template=ranking_prompt_template,
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": ranking_output_parser.get_format_instructions()
            },
        )

        self.ranking_chain = ranking_prompt | self.llm | ranking_output_parser

    def analyze_documents(
        self, query: str, documents: List[Dict], analysis_type: str = "best_match"
    ) -> Dict:
        """
        Analyze documents using LLM.

        Args:
            query: The search query
            documents: List of documents to analyze
            analysis_type: Type of analysis ("best_match" or "rank_all")

        Returns:
            Dict containing analysis results
        """
        if not self.llm:
            logger.error("LLM not initialized")
            return {"error": "LLM not initialized"}

        try:
            # Format documents into context string
            context_parts = []
            for doc in documents:
                # Normalize document fields
                normalized_doc = {}
                for field, value in doc.items():
                    if field == "_id":
                        normalized_doc[field] = str(value)
                    elif field in ["skills", "may_also_known_skills", "labels"]:
                        normalized_doc[field] = self._normalize_list_field(value)
                    else:
                        normalized_doc[field] = self._normalize_field_value(value)

                context_parts.append(
                    json.dumps(normalized_doc, indent=2, cls=JSONEncoder)
                )

            context_string = "\n\n---\n\n".join(context_parts)

            # Perform analysis based on type
            if analysis_type == "best_match":
                result = self.retrieval_chain.invoke(
                    {"context": context_string, "question": query}
                )
            else:  # rank_all
                result = self.ranking_chain.invoke(
                    {"context": context_string, "question": query}
                )

            # Add analysis metadata
            if isinstance(result, dict):
                result["analysis_info"] = {
                    "type": analysis_type,
                    "documents_analyzed": len(documents),
                    "query": query,
                }

            return result

        except Exception as e:
            logger.error(f"Error during document analysis: {e}")
            return {"error": str(e)}

    def _normalize_field_value(self, value) -> str:
        """Normalize field values for consistent processing."""
        if value is None:
            return ""
        if isinstance(value, (list, dict)):
            return json.dumps(value, cls=JSONEncoder)
        return str(value).strip()

    def _normalize_list_field(self, value) -> List[str]:
        """Normalize list fields for consistent processing."""
        if not value:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [self._normalize_field_value(item) for item in value if item]
        return [str(value)]


# Global wrapper functions for the new classes
def get_document_retriever(
    mongo_uri: str = None, db_name: str = None, collection_name: str = None
) -> DocumentRetriever:
    """Get or create a DocumentRetriever instance."""
    global _document_retriever
    if _document_retriever is None:
        mongo_uri = mongo_uri or MONGODB_URI
        db_name = db_name or DB_NAME
        collection_name = collection_name or COLLECTION_NAME
        _document_retriever = DocumentRetriever(mongo_uri, db_name, collection_name)
    return _document_retriever


def get_llm_analyzer(groq_api_key: str = None) -> LLMAnalyzer:
    """Get or create an LLMAnalyzer instance."""
    global _llm_analyzer
    if _llm_analyzer is None:
        groq_api_key = groq_api_key or GROQ_API_KEY
        _llm_analyzer = LLMAnalyzer(groq_api_key)
    return _llm_analyzer


# Global instances
_document_retriever = None
_llm_analyzer = None


# Example usage functions
def search_documents(query: str, limit: int = 50) -> Dict:
    """Search for documents using vector similarity."""
    retriever = get_document_retriever()
    return retriever.retrieve_documents(query, limit)


def analyze_documents(
    query: str, document_ids: List[str], analysis_type: str = "best_match"
) -> Dict:
    """Analyze specific documents using LLM."""
    retriever = get_document_retriever()
    analyzer = get_llm_analyzer()

    # First get document details
    doc_details = retriever.get_document_details(document_ids)
    if "error" in doc_details:
        return doc_details

    # Then analyze the documents
    return analyzer.analyze_documents(query, doc_details["documents"], analysis_type)
