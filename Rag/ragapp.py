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
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Core modules from the codebase
from core.custom_logger import CustomLogger
from core.config import AppConfig
from core.helpers import JSONEncoder
from core.llm_factory import LLMFactory  # Use centralized LLM factory
from core.llm_config import LLMConfigManager, LLMProvider
from core.exceptions import LLMProviderError
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
    "MONGO_URI", AppConfig.MONGODB_URI or "YOUR_MONGO_URI_HERE"
)
DB_NAME = os.environ.get("DB_NAME", AppConfig.DB_NAME or "YOUR_DB_NAME_HERE")
COLLECTION_NAME = os.environ.get(
    "COLLECTION_NAME", AppConfig.COLLECTION_NAME or "YOUR_COLLECTION_NAME_HERE"
)
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
        self.ranking_chain = None
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
        """Initialize LLM using centralized factory"""
        try:
            # Initialize LLM configuration manager
            self.llm_manager = LLMConfigManager()

            # Use the centralized LLM factory which handles provider selection
            self.llm = LLMFactory.create_llm()

            provider_name = self.llm_manager.provider.value
            logger.info(f"RAGApp LLM initialized using {provider_name} provider")

        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise LLMProviderError(f"Failed to initialize RAGApp LLM: {e}")

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
    ) -> Tuple[List[ObjectId], Optional[str], Dict[str, int], List[Dict]]:
        """
        Enhanced version with separate control for MongoDB retrieval and LLM context.

        Args:
            question: The search query
            mongodb_retrieval_limit: Maximum documents to retrieve from MongoDB (default: 50)
            llm_context_limit: Maximum documents to send to LLM (default: 10)

        Returns:
            Tuple of (document_ids, context_string, statistics, scored_documents)
            where statistics contains counts of documents at each stage
            and scored_documents contains the sorted list of documents with their scores
        """
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot retrieve documents.")
            return [], None, {"mongodb_retrieved": 0, "llm_context_sent": 0}, []

        # Maximum context length for LLM (adjust based on your model's limits)
        MAX_CONTEXT_LENGTH = 8000  # Conservative limit for most LLMs

        logger.info(
            f"Retrieving up to {mongodb_retrieval_limit} documents from MongoDB for question: {question}"
        )
        logger.info(
            f"Will send up to {llm_context_limit} documents to LLM for processing"
        )

        try:
            # Step 1: Retrieve documents from MongoDB using vector search with scores
            retrieved_docs_with_score = self.vector_store.similarity_search_with_score(
                query=question, k=mongodb_retrieval_limit
            )

            if not retrieved_docs_with_score:
                logger.warning("No relevant documents found in MongoDB.")
                return [], None, {"mongodb_retrieved": 0, "llm_context_sent": 0}, []

            # Extract _ids and scores from metadata
            scored_documents = []
            doc_ids = []
            for doc, score in retrieved_docs_with_score:
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

                    # Store document with its score
                    scored_documents.append(
                        {
                            "_id": str(doc_id),
                            "score": score,
                            "text_preview": (
                                doc.page_content[:200] + "..."
                                if len(doc.page_content) > 200
                                else doc.page_content
                            ),
                        }
                    )
                    doc_ids.append(doc_id)
                else:
                    logger.warning(f"Retrieved document missing _id in metadata: {doc}")

            if not doc_ids:
                logger.error("Could not extract valid _ids from retrieved documents.")
                return [], None, {"mongodb_retrieved": 0, "llm_context_sent": 0}, []

            # Sort documents by score in descending order
            scored_documents.sort(key=lambda x: x["score"], reverse=True)

            # Log the sorted results
            logger.info("=== SORTED DOCUMENTS BY RELEVANCE SCORE ===")
            for i, doc in enumerate(scored_documents):
                logger.info(f"Rank {i+1}: ID: {doc['_id']}, Score: {doc['score']:.4f}")
                logger.info(f"Preview: {doc['text_preview']}")
            logger.info("=== END OF SORTED DOCUMENTS ===")

            mongodb_count = len(doc_ids)
            logger.info(
                f"Successfully retrieved {mongodb_count} documents from MongoDB"
            )

            # Step 2: Limit documents for LLM context based on sorted scores
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
            total_context_length = 0
            actual_llm_count = 0

            logger.info("=== DOCUMENTS BEING SENT TO LLM ===")
            for i, doc in enumerate(fetched_docs_cursor):
                # Normalize document fields
                normalized_doc = {}
                for field, value in doc.items():
                    if field == "_id":
                        normalized_doc[field] = str(value)
                    elif field in ["skills", "may_also_known_skills", "labels"]:
                        normalized_doc[field] = self._normalize_list_field(value)
                    else:
                        normalized_doc[field] = self._normalize_field_value(value)

                # Convert to JSON string and check length
                doc_json = json.dumps(normalized_doc, indent=2, cls=JSONEncoder)
                doc_length = len(doc_json)

                # Check if adding this document would exceed the context limit
                if total_context_length + doc_length > MAX_CONTEXT_LENGTH:
                    logger.warning(
                        f"Context length limit reached after {actual_llm_count} documents"
                    )
                    break

                context_parts.append(doc_json)
                total_context_length += doc_length
                actual_llm_count += 1

                logger.info(
                    f"LLM Document {actual_llm_count}/{llm_count} added to context"
                )
                logger.info(
                    f"Current context length: {total_context_length} characters"
                )

            logger.info("=== END OF LLM DOCUMENTS ===")

            if not context_parts:
                logger.error(
                    "No documents could be added to context due to length constraints"
                )
                return (
                    doc_ids,
                    None,
                    {
                        "mongodb_retrieved": mongodb_count,
                        "llm_context_sent": 0,
                        "context_length": 0,
                        "context_limit_exceeded": True,
                    },
                    scored_documents,
                )

            context_string = "\n\n---\n\n".join(context_parts)

            # Print the context statistics
            logger.info("=== CONTEXT STATISTICS ===")
            logger.info(f"Documents retrieved from MongoDB: {mongodb_count}")
            logger.info(f"Documents sent to LLM: {actual_llm_count}")
            logger.info(f"Context length: {len(context_string)} characters")
            logger.info(f"Context length limit: {MAX_CONTEXT_LENGTH} characters")
            logger.info("=== END OF STATISTICS ===")

            statistics = {
                "mongodb_retrieved": mongodb_count,
                "llm_context_sent": actual_llm_count,
                "context_length": len(context_string),
                "context_limit_exceeded": len(context_string) >= MAX_CONTEXT_LENGTH,
            }

            return doc_ids, context_string, statistics, scored_documents

        except Exception as e:
            logger.error(f"Error during enhanced retrieval or context preparation: {e}")
            return [], None, {"mongodb_retrieved": 0, "llm_context_sent": 0}, []

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
            Dict containing the best matching document ID, processing statistics, and scored documents
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

        retrieved_ids, context, statistics, scored_documents = (
            self.get_relevant_ids_and_context_enhanced(
                question, mongodb_retrieval_limit, llm_context_limit
            )
        )

        if context is None:
            if statistics.get("context_limit_exceeded", False):
                logger.error(
                    "Context length limit exceeded. No documents could be processed by LLM."
                )
                return {
                    "error": "Context length limit exceeded",
                    "statistics": statistics,
                    "scored_documents": scored_documents,
                    "message": "The retrieved documents were too large to process. Consider reducing the number of documents or their content size.",
                }
            else:
                logger.error("Failed to get context or retrieve IDs.")
                return {
                    "error": "No documents found or context preparation failed",
                    "statistics": statistics,
                    "scored_documents": scored_documents,
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

                # Add processing statistics and scored documents
                result["statistics"] = statistics
                result["scored_documents"] = scored_documents
                result["processing_info"] = {
                    "mongodb_retrieval_limit": mongodb_retrieval_limit,
                    "llm_context_limit": llm_context_limit,
                    "documents_considered": statistics["mongodb_retrieved"],
                    "documents_analyzed_by_llm": statistics["llm_context_sent"],
                    "context_length": statistics["context_length"],
                    "context_limit_exceeded": statistics.get(
                        "context_limit_exceeded", False
                    ),
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
                    "scored_documents": scored_documents,
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
                        "scored_documents": scored_documents,
                        "fallback": True,
                    }
            return {
                "error": str(e),
                "statistics": statistics,
                "scored_documents": scored_documents,
            }

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
            Dict containing candidate IDs, processing statistics, and scored documents
        """
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot retrieve documents.")
            return None

        logger.info(
            f"Retrieving candidates with limits - MongoDB: {mongodb_retrieval_limit}, LLM: {llm_context_limit}"
        )

        try:
            # Get relevant documents and their IDs with enhanced limits
            retrieved_ids, context, statistics, scored_documents = (
                self.get_relevant_ids_and_context_enhanced(
                    question, mongodb_retrieval_limit, llm_context_limit
                )
            )

            if not retrieved_ids:
                logger.warning("No relevant documents found.")
                return {
                    "message": "No candidates found",
                    "statistics": statistics,
                    "scored_documents": scored_documents,
                }

            # Create the result dictionary with all retrieved IDs
            result = {
                "total_mongodb_retrieved": statistics["mongodb_retrieved"],
                "total_llm_processed": statistics["llm_context_sent"],
                "processing_info": {
                    "mongodb_retrieval_limit": mongodb_retrieval_limit,
                    "llm_context_limit": llm_context_limit,
                },
                "scored_documents": scored_documents[
                    :max_results
                ],  # Limit to max_results
            }

            # Add all retrieved IDs (from MongoDB)
            for i, doc_id in enumerate(retrieved_ids[:max_results]):
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
            Dict containing all ranked candidates with processing statistics and scored documents
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
            retrieved_ids, context, statistics, scored_documents = (
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
                    "scored_documents": scored_documents,
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
                        "scored_documents": scored_documents,
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
                        "scored_documents": scored_documents,
                    }
            else:
                logger.error(f"Unexpected ranking result type: {type(result)}")
                return {
                    "error": "Unexpected ranking result type",
                    "raw_result": str(result),
                    "statistics": statistics,
                    "scored_documents": scored_documents,
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
                "scored_documents": (
                    scored_documents if "scored_documents" in locals() else []
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

    def vector_similarity_search(self, query: str, limit: int = 50) -> Dict:
        """
        Perform pure vector similarity search and return results sorted by score.

        Args:
            query: The search query
            limit: Maximum number of documents to retrieve (default: 50)

        Returns:
            Dict containing:
            - results: List of documents with scores
            - total_found: Total number of documents found
            - statistics: Search statistics
        """
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot perform search.")
            return {"error": "Vector store not initialized"}

        try:
            logger.info(f"Performing vector similarity search for: {query}")
            logger.info(f"Retrieval limit: {limit}")

            # Perform similarity search with scores
            results_with_scores = self.vector_store.similarity_search_with_score(
                query=query, k=limit
            )

            if not results_with_scores:
                logger.warning("No documents found matching the query.")
                return {"results": [], "total_found": 0, "statistics": {"retrieved": 0}}

            # Process and sort results
            processed_results = []
            for doc, score in results_with_scores:
                if hasattr(doc, "metadata") and "_id" in doc.metadata:
                    doc_id = doc.metadata["_id"]

                    # Fetch the complete document from MongoDB
                    complete_doc = self.collection.find_one({"_id": ObjectId(doc_id)})

                    if complete_doc:
                        # Format the document according to the specified structure
                        formatted_doc = {
                            "_id": str(complete_doc.get("_id", "")),
                            "user_id": str(complete_doc.get("user_id", "")),
                            "username": str(complete_doc.get("username", "")),
                            "contact_details": {
                                "name": str(
                                    complete_doc.get("contact_details", {}).get(
                                        "name", ""
                                    )
                                ),
                                "email": str(
                                    complete_doc.get("contact_details", {}).get(
                                        "email", ""
                                    )
                                ),
                                "phone": str(
                                    complete_doc.get("contact_details", {}).get(
                                        "phone", ""
                                    )
                                ),
                                "alternative_phone": str(
                                    complete_doc.get("contact_details", {}).get(
                                        "alternative_phone", ""
                                    )
                                ),
                                "current_city": str(
                                    complete_doc.get("contact_details", {}).get(
                                        "current_city", ""
                                    )
                                ),
                                "looking_for_jobs_in": complete_doc.get(
                                    "contact_details", {}
                                ).get("looking_for_jobs_in", []),
                                "gender": str(
                                    complete_doc.get("contact_details", {}).get(
                                        "gender", ""
                                    )
                                ),
                                "age": int(
                                    complete_doc.get("contact_details", {}).get(
                                        "age", 0
                                    )
                                ),
                                "naukri_profile": str(
                                    complete_doc.get("contact_details", {}).get(
                                        "naukri_profile", ""
                                    )
                                ),
                                "linkedin_profile": str(
                                    complete_doc.get("contact_details", {}).get(
                                        "linkedin_profile", ""
                                    )
                                ),
                                "portfolio_link": str(
                                    complete_doc.get("contact_details", {}).get(
                                        "portfolio_link", ""
                                    )
                                ),
                                "pan_card": str(
                                    complete_doc.get("contact_details", {}).get(
                                        "pan_card", ""
                                    )
                                ),
                                "aadhar_card": str(
                                    complete_doc.get("contact_details", {}).get(
                                        "aadhar_card", ""
                                    )
                                ),
                            },
                            "total_experience": str(
                                complete_doc.get("total_experience", "")
                            ),
                            "notice_period": str(complete_doc.get("notice_period", "")),
                            "currency": str(complete_doc.get("currency", "")),
                            "pay_duration": str(complete_doc.get("pay_duration", "")),
                            "current_salary": float(
                                complete_doc.get("current_salary", 0)
                            ),
                            "hike": float(complete_doc.get("hike", 0)),
                            "expected_salary": float(
                                complete_doc.get("expected_salary", 0)
                            ),
                            "skills": complete_doc.get("skills", []),
                            "may_also_known_skills": complete_doc.get(
                                "may_also_known_skills", []
                            ),
                            "labels": complete_doc.get("labels", []),
                            "experience": complete_doc.get("experience", []),
                            "academic_details": complete_doc.get(
                                "academic_details", []
                            ),
                            "source": str(complete_doc.get("source", "")),
                            "last_working_day": str(
                                complete_doc.get("last_working_day", "")
                            ),
                            "is_tier1_mba": bool(
                                complete_doc.get("is_tier1_mba", False)
                            ),
                            "is_tier1_engineering": bool(
                                complete_doc.get("is_tier1_engineering", False)
                            ),
                            "comment": str(complete_doc.get("comment", "")),
                            "exit_reason": str(complete_doc.get("exit_reason", "")),
                            "similarity_score": float(score),
                        }
                        processed_results.append(formatted_doc)

            # Sort by score in descending order
            processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Log results
            logger.info("=== VECTOR SEARCH RESULTS ===")
            for i, result in enumerate(processed_results):
                logger.info(
                    f"Rank {i+1}: ID: {result['_id']}, Score: {result['similarity_score']:.4f}"
                )
            logger.info("=== END OF RESULTS ===")

            return {
                "results": processed_results,
                "total_found": len(processed_results),
                "statistics": {"retrieved": len(processed_results), "query": query},
            }

        except Exception as e:
            logger.error(f"Error during vector similarity search: {e}")
            return {"error": str(e)}

    def llm_context_search(self, query: str, context_size: int = 5) -> Dict:
        """
        Perform LLM-based search with user-controlled context size.

        Args:
            query: The search query
            context_size: Number of documents to include in LLM context (default: 5)

        Returns:
            Dict containing:
            - results: List of documents with relevance scores and explanations
            - total_analyzed: Number of documents analyzed by LLM
            - statistics: Search statistics
        """
        if not self.ranking_chain:
            logger.error("Ranking chain not initialized. Cannot perform LLM search.")
            return {"error": "Ranking chain not initialized"}

        try:
            logger.info(f"Performing LLM search for: {query}")
            logger.info(f"Context size: {context_size}")

            # First get documents using vector search
            vector_results = self.vector_similarity_search(query, limit=context_size)

            if "error" in vector_results:
                return vector_results

            if not vector_results["results"]:
                return {
                    "results": [],
                    "total_analyzed": 0,
                    "statistics": {"retrieved": 0, "analyzed": 0},
                }

            # Get document IDs for context
            doc_ids = [result["_id"] for result in vector_results["results"]]

            # Fetch full documents
            projection = {field: 1 for field in FIELDS_TO_EXTRACT}
            if "_id" not in projection:
                projection["_id"] = 1

            fetched_docs = list(
                self.collection.find(
                    {"_id": {"$in": [ObjectId(doc_id) for doc_id in doc_ids]}},
                    projection,
                )
            )

            # Format context
            context_parts = []
            for doc in fetched_docs:
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

            # Use LLM to analyze and rank documents
            logger.info("Invoking LLM for document analysis...")
            result = self.ranking_chain.invoke(
                {"context": context_string, "question": query}
            )

            if isinstance(result, dict) and "matches" in result:
                # Format results with explanations
                formatted_results = []
                for match in result["matches"]:
                    if isinstance(match, dict) and "_id" in match:
                        # Fetch the complete document
                        complete_doc = self.collection.find_one(
                            {"_id": ObjectId(match["_id"])}
                        )

                        if complete_doc:
                            formatted_doc = {
                                "_id": str(complete_doc.get("_id", "")),
                                "user_id": str(complete_doc.get("user_id", "")),
                                "username": str(complete_doc.get("username", "")),
                                "contact_details": {
                                    "name": str(
                                        complete_doc.get("contact_details", {}).get(
                                            "name", ""
                                        )
                                    ),
                                    "email": str(
                                        complete_doc.get("contact_details", {}).get(
                                            "email", ""
                                        )
                                    ),
                                    "phone": str(
                                        complete_doc.get("contact_details", {}).get(
                                            "phone", ""
                                        )
                                    ),
                                    "alternative_phone": str(
                                        complete_doc.get("contact_details", {}).get(
                                            "alternative_phone", ""
                                        )
                                    ),
                                    "current_city": str(
                                        complete_doc.get("contact_details", {}).get(
                                            "current_city", ""
                                        )
                                    ),
                                    "looking_for_jobs_in": complete_doc.get(
                                        "contact_details", {}
                                    ).get("looking_for_jobs_in", []),
                                    "gender": str(
                                        complete_doc.get("contact_details", {}).get(
                                            "gender", ""
                                        )
                                    ),
                                    "age": int(
                                        complete_doc.get("contact_details", {}).get(
                                            "age", 0
                                        )
                                    ),
                                    "naukri_profile": str(
                                        complete_doc.get("contact_details", {}).get(
                                            "naukri_profile", ""
                                        )
                                    ),
                                    "linkedin_profile": str(
                                        complete_doc.get("contact_details", {}).get(
                                            "linkedin_profile", ""
                                        )
                                    ),
                                    "portfolio_link": str(
                                        complete_doc.get("contact_details", {}).get(
                                            "portfolio_link", ""
                                        )
                                    ),
                                    "pan_card": str(
                                        complete_doc.get("contact_details", {}).get(
                                            "pan_card", ""
                                        )
                                    ),
                                    "aadhar_card": str(
                                        complete_doc.get("contact_details", {}).get(
                                            "aadhar_card", ""
                                        )
                                    ),
                                },
                                "total_experience": str(
                                    complete_doc.get("total_experience", "")
                                ),
                                "notice_period": str(
                                    complete_doc.get("notice_period", "")
                                ),
                                "currency": str(complete_doc.get("currency", "")),
                                "pay_duration": str(
                                    complete_doc.get("pay_duration", "")
                                ),
                                "current_salary": float(
                                    complete_doc.get("current_salary", 0)
                                ),
                                "hike": float(complete_doc.get("hike", 0)),
                                "expected_salary": float(
                                    complete_doc.get("expected_salary", 0)
                                ),
                                "skills": complete_doc.get("skills", []),
                                "may_also_known_skills": complete_doc.get(
                                    "may_also_known_skills", []
                                ),
                                "labels": complete_doc.get("labels", []),
                                "experience": complete_doc.get("experience", []),
                                "academic_details": complete_doc.get(
                                    "academic_details", []
                                ),
                                "source": str(complete_doc.get("source", "")),
                                "last_working_day": str(
                                    complete_doc.get("last_working_day", "")
                                ),
                                "is_tier1_mba": bool(
                                    complete_doc.get("is_tier1_mba", False)
                                ),
                                "is_tier1_engineering": bool(
                                    complete_doc.get("is_tier1_engineering", False)
                                ),
                                "comment": str(complete_doc.get("comment", "")),
                                "exit_reason": str(complete_doc.get("exit_reason", "")),
                                "relevance_score": float(
                                    match.get("relevance_score", 0.0)
                                ),
                                "match_reason": str(
                                    match.get("match_reason", "No explanation provided")
                                ),
                            }
                            formatted_results.append(formatted_doc)

                # Sort by relevance score
                formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)

                logger.info("=== LLM SEARCH RESULTS ===")
                for i, result in enumerate(formatted_results):
                    logger.info(f"Rank {i+1}: ID: {result['_id']}")
                    logger.info(f"Score: {result['relevance_score']:.4f}")
                    logger.info(f"Explanation: {result['match_reason']}")
                logger.info("=== END OF RESULTS ===")

                return {
                    "results": formatted_results,
                    "total_analyzed": len(formatted_results),
                    "statistics": {
                        "retrieved": len(vector_results["results"]),
                        "analyzed": len(formatted_results),
                        "query": query,
                    },
                }
            else:
                logger.error(f"Unexpected LLM result format: {result}")
                return {
                    "error": "Unexpected LLM result format",
                    "raw_result": str(result),
                }

        except Exception as e:
            logger.error(f"Error during LLM search: {e}")
            return {"error": str(e)}
