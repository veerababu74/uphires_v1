import os
import json
import time
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

# Initialize logger
logger_instance = CustomLogger()
logger = logger_instance.get_logger("rag_application")

# Load environment variables
load_dotenv()

# Configuration with fallbacks
MONGODB_URI = os.environ.get(
    "MONGO_URI", AppConfig.MONGODB_CONNECTION_STRING or "YOUR_MONGO_URI_HERE"
)
DB_NAME = os.environ.get("DB_NAME", "YOUR_DB_NAME_HERE")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "YOUR_COLLECTION_NAME_HERE")
VECTOR_FIELD = os.environ.get("VECTOR_FIELD", "combined_resume_vector")
INDEX_NAME = "vector_search_index"
GROQ_API_KEY = os.environ.get("GROQ_API_KEYS", "").split(",")[0].strip()

# Fields to extract from retrieved documents
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
        populate_by_name = True


class DocumentRetriever:
    """Class responsible for retrieving documents from MongoDB using vector search."""

    def __init__(
        self, mongo_uri: str = None, db_name: str = None, collection_name: str = None
    ):
        """Initialize the document retriever with MongoDB connection details."""
        self.client = None
        self.collection = None
        self.vector_store = None
        self.embeddings = None
        self.vectorizer = None

        # Use provided values or defaults
        self.mongo_uri = mongo_uri or MONGODB_URI
        self.db_name = db_name or DB_NAME
        self.collection_name = collection_name or COLLECTION_NAME

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize MongoDB connection and vector store."""
        try:
            # Initialize vectorizer
            self.vectorizer = Vectorizer()
            self.embeddings = VectorizerEmbeddingAdapter(self.vectorizer)

            # Initialize MongoDB connection
            self.client = MongoClient(self.mongo_uri)
            db = self.client[self.db_name]
            self.collection = db[self.collection_name]
            self.client.admin.command("ping")

            # Initialize vector store
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name=INDEX_NAME,
                text_key="combined_resume",
                embedding_key=VECTOR_FIELD,
            )

            logger.info(
                f"DocumentRetriever initialized successfully for {self.db_name}.{self.collection_name}"
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
                    "search_time": time.time(),
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
        if self.collection is None:
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

    def __init__(self, groq_api_key: str = None):
        """Initialize the LLM analyzer with Groq API key."""
        self.llm = None
        self.retrieval_chain = None
        self.groq_api_key = groq_api_key or GROQ_API_KEY

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize LLM and analysis chains."""
        try:
            # Initialize Groq LLM
            self.llm = ChatGroq(
                api_key=self.groq_api_key, model="gemma2-9b-it", temperature=0.0
            )

            # Setup retrieval chain
            self._setup_retrieval_chain()

            logger.info("LLMAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLMAnalyzer: {e}")
            raise

    def _setup_retrieval_chain(self):
        """Setup the retrieval chain for finding best matching document."""
        prompt_template_text = """You are an expert AI recruiter analyzing candidate resumes to find the best match for a specific job requirement.

Your task is to analyze the following candidate profiles and identify the single best matching candidate based on the user's query.

For each candidate, you have access to:
1. Basic Information (username, contact details)
2. Experience Details (total experience, notice period)
3. Salary Information (current salary, expected salary, currency)
4. Skills (primary skills and additional skills)
5. Academic Background
6. Other relevant details

CRITICAL INSTRUCTIONS:
1. Carefully analyze each candidate's profile
2. Consider all aspects: skills, experience, salary expectations, notice period
3. Compare each candidate against the query requirements
4. Select the single best matching candidate
5. Return ONLY the MongoDB '_id' of the best matching candidate in JSON format

{format_instructions}

Candidate Profiles:
---
{context}
---

User Query: {question}

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

    def analyze_documents(self, query: str, documents: List[Dict]) -> Dict:
        """
        Analyze documents using LLM to find the best match.

        Args:
            query: The search query
            documents: List of documents to analyze

        Returns:
            Dict containing analysis results
        """
        if not self.llm:
            logger.error("LLM not initialized")
            return {"error": "LLM not initialized"}

        try:
            # Format documents into context string with comprehensive information
            context_parts = []
            for doc in documents:
                # Create a structured format for each document
                formatted_doc = {
                    "Basic Information": {
                        "ID": str(doc.get("_id", "N/A")),
                        "Username": doc.get("username", "N/A"),
                        "Contact Details": doc.get("contact_details", "N/A"),
                    },
                    "Experience": {
                        "Total Experience": doc.get("total_experience", "N/A"),
                        "Notice Period": doc.get("notice_period", "N/A"),
                        "Last Working Day": doc.get("last_working_day", "N/A"),
                    },
                    "Salary Details": {
                        "Current Salary": f"{doc.get('current_salary', 'N/A')} {doc.get('currency', '')}",
                        "Expected Salary": f"{doc.get('expected_salary', 'N/A')} {doc.get('currency', '')}",
                        "Expected Hike": doc.get("hike", "N/A"),
                    },
                    "Skills": {
                        "Primary Skills": doc.get("skills", ["N/A"]),
                        "Additional Skills": doc.get("may_also_known_skills", ["N/A"]),
                        "Labels": doc.get("labels", ["N/A"]),
                    },
                    "Education": {
                        "Academic Details": doc.get("academic_details", "N/A"),
                        "Tier 1 MBA": doc.get("is_tier1_mba", False),
                        "Tier 1 Engineering": doc.get("is_tier1_engineering", False),
                    },
                    "Experience Details": doc.get("experience", "N/A"),
                }

                # Convert to formatted string
                context_parts.append(
                    f"CANDIDATE PROFILE:\n{json.dumps(formatted_doc, indent=2, cls=JSONEncoder)}"
                )

            context_string = "\n\n---\n\n".join(context_parts)

            # Log the context being sent to LLM
            logger.info("Sending context to LLM:")
            logger.info(context_string)

            # Perform analysis
            result = self.retrieval_chain.invoke(
                {"context": context_string, "question": query}
            )

            # Add analysis metadata
            if isinstance(result, dict):
                result["analysis_info"] = {
                    "documents_analyzed": len(documents),
                    "query": query,
                    "context_length": len(context_string),
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


# Helper class for vectorizer adapter
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


# Example usage
def main():
    try:
        # Initialize components
        retriever = DocumentRetriever()
        analyzer = LLMAnalyzer()

        # Example query
        query = "Find Python developers with machine learning experience"
        limit = 4

        print("\n🔍 Starting document retrieval process...")
        print(f"Query: {query}")
        print(f"Retrieval limit: {limit} documents")

        # Step 1: Retrieve documents
        retrieval_result = retriever.retrieve_documents(query, limit)
        if "error" in retrieval_result:
            print(f"❌ Error during retrieval: {retrieval_result['error']}")
            return

        if not retrieval_result["documents"]:
            print("❌ No documents found matching the query")
            return

        # Print detailed retrieval results
        print("\n📄 Retrieved Documents:")
        print("=" * 80)
        for i, doc in enumerate(retrieval_result["documents"], 1):
            print(f"\nDocument {i}:")
            print(f"ID: {doc['_id']}")
            print(f"Similarity Score: {doc['score']:.4f}")
            print(f"Text Preview: {doc['text_preview']}")
            print("-" * 80)

        # Step 2: Get full document details
        print("\n🔎 Fetching full document details...")
        doc_ids = [doc["_id"] for doc in retrieval_result["documents"]]
        doc_details = retriever.get_document_details(doc_ids)
        if "error" in doc_details:
            print(f"❌ Error getting document details: {doc_details['error']}")
            return

        if not doc_details["documents"]:
            print("❌ No document details found for the retrieved IDs")
            return

        # Print detailed document information
        print("\n📋 Full Document Details:")
        print("=" * 80)
        for i, doc in enumerate(doc_details["documents"], 1):
            print(f"\nDocument {i} Details:")
            print(f"ID: {doc['_id']}")
            print(f"Username: {doc.get('username', 'N/A')}")
            print(f"Total Experience: {doc.get('total_experience', 'N/A')}")
            print(f"Skills: {', '.join(doc.get('skills', ['N/A']))}")
            print(
                f"Current Salary: {doc.get('current_salary', 'N/A')} {doc.get('currency', '')}"
            )
            print(
                f"Expected Salary: {doc.get('expected_salary', 'N/A')} {doc.get('currency', '')}"
            )
            print(f"Notice Period: {doc.get('notice_period', 'N/A')}")
            print("-" * 80)

        # Step 3: Analyze documents with LLM
        print("\n🤖 Analyzing documents with LLM...")
        analysis_result = analyzer.analyze_documents(query, doc_details["documents"])
        if "error" in analysis_result:
            print(f"❌ Error during analysis: {analysis_result['error']}")
            return

        # Print analysis results
        print("\n✨ Analysis Results:")
        print("=" * 80)
        print(f"Best matching document ID: {analysis_result.get('_id')}")
        if "analysis_info" in analysis_result:
            print(
                f"Documents analyzed: {analysis_result['analysis_info']['documents_analyzed']}"
            )
        print("=" * 80)

    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
