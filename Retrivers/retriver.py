import os
import json
from typing import List, Dict, Optional
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

# Import the internal vectorizer for embeddings
from embeddings.vectorizer import Vectorizer
from core.custom_logger import CustomLogger
from core.config import AppConfig
from core.helpers import JSONEncoder

# LangChain imports
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Initialize logger
logger_instance = CustomLogger()
logger = logger_instance.get_logger("rag_application")

# Fields to extract from MongoDB documents
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


class MangoRetriever:
    """RAG Application for retrieving and ranking documents from MongoDB."""

    def __init__(self):
        """Initialize the RAG Retriever with MongoDB connection and vectorizer."""
        self.client = None
        self.collection = None
        self.vectorizer = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize MongoDB connection and vectorizer."""
        try:
            # Get MongoDB connection details from AppConfig
            from core.config import AppConfig

            mongo_uri = AppConfig.MONGODB_URI
            db_name = AppConfig.DB_NAME
            collection_name = AppConfig.COLLECTION_NAME

            # Initialize vectorizer
            self.vectorizer = Vectorizer()
            logger.info("Vectorizer initialized successfully")

            # Initialize MongoDB connection
            self.client = MongoClient(mongo_uri)
            db = self.client[db_name]
            self.collection = db[collection_name]
            self.client.admin.command("ping")
            logger.info(
                f"Connected to MongoDB database '{db_name}' and collection '{collection_name}'"
            )

        except Exception as e:
            logger.error(f"Failed to initialize RAG Retriever: {e}")
            raise

    def search_and_rank(self, question: str, limit: int = 10) -> Dict:
        """
        Search MongoDB using vector similarity and return ranked results.

        Args:
            question (str): The search query/question
            limit (int): Maximum number of results to return

        Returns:
            Dict containing:
            - results: List of ranked documents with scores
            - total_count: Total number of results
            - query: Original query
        """
        try:
            # Generate embedding for the question
            question_embedding = self.vectorizer.generate_embedding(question)

            # Create projection for all fields we want to extract
            projection = {field: 1 for field in FIELDS_TO_EXTRACT}
            projection["score"] = {"$meta": "vectorSearchScore"}

            # Perform vector similarity search using MongoDB's $vectorSearch
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_search_index",
                        "path": "combined_resume_vector",
                        "queryVector": question_embedding,
                        "numCandidates": limit
                        * 2,  # Get more candidates for better ranking
                        "limit": limit,
                    }
                },
                {"$project": projection},
            ]

            # Execute the search
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Retrieved {len(results)} documents from MongoDB")

            # Format and normalize results
            formatted_results = []
            for doc in results:
                try:
                    # Ensure _id is properly handled
                    doc_id = doc.get("_id")
                    if isinstance(doc_id, ObjectId):
                        doc_id = str(doc_id)
                    elif doc_id is None:
                        logger.warning("Document missing _id field")
                        continue

                    formatted_doc = {
                        "_id": doc_id,
                        "score": float(doc.get("score", 0.0)),
                    }

                    # Add all other fields with proper handling
                    for field in FIELDS_TO_EXTRACT:
                        if field != "_id":  # _id already handled above
                            value = doc.get(field)
                            if isinstance(value, (list, dict)):
                                formatted_doc[field] = value
                            elif value is None:
                                formatted_doc[field] = (
                                    ""
                                    if field
                                    in ["username", "contact_details", "source"]
                                    else 0
                                )
                            else:
                                formatted_doc[field] = value

                    formatted_results.append(formatted_doc)
                    logger.debug(f"Formatted document with ID: {doc_id}")

                except Exception as e:
                    logger.error(f"Error formatting document: {e}")
                    continue

            # Sort results by score in descending order
            sorted_results = sorted(
                formatted_results, key=lambda x: x["score"], reverse=True
            )
            logger.info(
                f"Successfully formatted and sorted {len(sorted_results)} results"
            )

            return {
                "results": sorted_results,
                "total_count": len(sorted_results),
                "query": question,
            }

        except Exception as e:
            logger.error(f"Error during search and rank: {e}")
            return {"error": str(e), "results": [], "total_count": 0, "query": question}


class LangChainRetriever:
    """RAG Application using LangChain for retrieving and ranking documents from MongoDB."""

    def __init__(self):
        """Initialize the LangChain Retriever with MongoDB connection and vectorizer."""
        self.client = None
        self.collection = None
        self.vectorizer = None
        self.vector_store = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize MongoDB connection, vectorizer, and LangChain components."""
        try:
            # Get MongoDB connection details from AppConfig
            mongo_uri = AppConfig.MONGODB_URI
            db_name = AppConfig.DB_NAME
            collection_name = AppConfig.COLLECTION_NAME

            # Initialize vectorizer and create adapter
            self.vectorizer = Vectorizer()
            embedding_adapter = VectorizerEmbeddingAdapter(self.vectorizer)
            logger.info("Vectorizer and embedding adapter initialized successfully")

            # Initialize MongoDB connection
            self.client = MongoClient(mongo_uri)
            db = self.client[db_name]
            self.collection = db[collection_name]
            self.client.admin.command("ping")
            logger.info(
                f"Connected to MongoDB database '{db_name}' and collection '{collection_name}'"
            )

            # Initialize LangChain vector store with the adapter
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=embedding_adapter,  # Use the adapter instead of vectorizer directly
                index_name="vector_search_index",
                text_key="combined_resume",
                embedding_key="combined_resume_vector",
            )
            logger.info("LangChain vector store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LangChain Retriever: {e}")
            raise

    def search_and_rank(self, question: str, limit: int = 10) -> Dict:
        """
        Search MongoDB using LangChain's vector similarity and return ranked results.

        Args:
            question (str): The search query/question
            limit (int): Maximum number of results to return

        Returns:
            Dict containing:
            - results: List of ranked documents with scores
            - total_count: Total number of results
            - query: Original query
        """
        try:
            # Use LangChain's similarity search with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=question, k=limit
            )
            logger.info(f"Retrieved {len(docs_with_scores)} documents using LangChain")

            # Format and normalize results
            formatted_results = []
            for doc, score in docs_with_scores:
                try:
                    # Get the document metadata
                    metadata = doc.metadata

                    # Ensure _id is properly handled
                    doc_id = metadata.get("_id")
                    if isinstance(doc_id, ObjectId):
                        doc_id = str(doc_id)
                    elif doc_id is None:
                        logger.warning("Document missing _id field")
                        continue

                    # Create base document with ID and score
                    formatted_doc = {"_id": doc_id, "score": float(score)}

                    # Add all fields from metadata
                    for field in FIELDS_TO_EXTRACT:
                        if field != "_id":  # _id already handled above
                            value = metadata.get(field)
                            if isinstance(value, (list, dict)):
                                formatted_doc[field] = value
                            elif value is None:
                                formatted_doc[field] = (
                                    ""
                                    if field
                                    in ["username", "contact_details", "source"]
                                    else 0
                                )
                            else:
                                formatted_doc[field] = value

                    formatted_results.append(formatted_doc)
                    logger.debug(f"Formatted document with ID: {doc_id}")

                except Exception as e:
                    logger.error(f"Error formatting document: {e}")
                    continue

            # Sort results by score in descending order
            sorted_results = sorted(
                formatted_results, key=lambda x: x["score"], reverse=True
            )
            logger.info(
                f"Successfully formatted and sorted {len(sorted_results)} results"
            )

            return {
                "results": sorted_results,
                "total_count": len(sorted_results),
                "query": question,
            }

        except Exception as e:
            logger.error(f"Error during LangChain search and rank: {e}")
            return {"error": str(e), "results": [], "total_count": 0, "query": question}

    def get_document_details(self, doc_id: str) -> Optional[Dict]:
        """
        Get detailed information for a specific document.

        Args:
            doc_id (str): The document ID to retrieve

        Returns:
            Optional[Dict]: Document details or None if not found
        """
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(doc_id)

            # Find the document
            doc = self.collection.find_one({"_id": object_id})

            if doc:
                # Convert ObjectId to string
                doc["_id"] = str(doc["_id"])
                return doc
            return None

        except Exception as e:
            logger.error(f"Error retrieving document details: {e}")
            return None


class RAGRetriever:
    """Retriever that combines vector search with RAG capabilities."""

    def __init__(self):
        self._initialize_components()

    def _initialize_components(self):
        """Initialize MongoDB connection and vectorizer."""
        try:
            # Get MongoDB connection details from AppConfig
            from core.config import AppConfig

            mongo_uri = AppConfig.MONGODB_URI
            db_name = AppConfig.DB_NAME
            collection_name = AppConfig.COLLECTION_NAME

            # Initialize vectorizer
            self.vectorizer = Vectorizer()
            logger.info("Vectorizer initialized successfully")

            # Initialize MongoDB connection
            self.client = MongoClient(mongo_uri)
            db = self.client[db_name]
            self.collection = db[collection_name]
            self.client.admin.command("ping")
            logger.info(
                f"Connected to MongoDB database '{db_name}' and collection '{collection_name}'"
            )

        except Exception as e:
            logger.error(f"Failed to initialize RAG Retriever: {e}")
            raise

    def search(self, query: str, limit: int = 5):
        """Search for relevant documents using RAG."""
        try:
            # Get query vector
            query_vector = self.vectorizer.get_embedding(query)

            # Perform vector search
            pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_vector,
                            "path": "combined_resume_vector",
                            "k": limit,
                        },
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "user_id": 1,
                        "username": 1,
                        "contact_details": 1,
                        "total_experience": 1,
                        "notice_period": 1,
                        "currency": 1,
                        "pay_duration": 1,
                        "current_salary": 1,
                        "hike": 1,
                        "expected_salary": 1,
                        "skills": 1,
                        "may_also_known_skills": 1,
                        "labels": 1,
                        "experience": 1,
                        "academic_details": 1,
                        "source": 1,
                        "last_working_day": 1,
                        "is_tier1_mba": 1,
                        "is_tier1_engineering": 1,
                        "score": {"$meta": "searchScore"},
                    }
                },
            ]

            results = list(self.collection.aggregate(pipeline))
            return results

        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            raise


def main():
    """Main function to demonstrate both retrievers."""
    try:
        # Initialize both retrievers
        rag_retriever = MangoRetriever()
        langchain_retriever = LangChainRetriever()
        print("✅ Both retrievers initialized successfully")

        while True:
            # Get user input
            question = input("\nEnter your search question (or 'quit' to exit): ")
            if question.lower() == "quit":
                break

            # Get limit from user
            try:
                limit = int(
                    input("Enter the number of results to return (default 10): ")
                    or "10"
                )
            except ValueError:
                limit = 10
                print("Invalid input, using default limit of 10")

            # Choose retriever
            retriever_choice = input("Choose retriever (1: RAG, 2: LangChain): ") or "1"
            retriever = (
                langchain_retriever if retriever_choice == "2" else rag_retriever
            )

            # Perform search and ranking
            print(f"\nSearching for: '{question}'")
            print(f"Limit: {limit} results")
            print(
                f"Using: {'LangChain' if retriever_choice == '2' else 'RAG'} Retriever"
            )
            print("-" * 50)

            results = retriever.search_and_rank(question, limit)

            if "error" in results:
                print(f"❌ Error: {results['error']}")
                continue

            # Display results
            print(f"\nFound {results['total_count']} results:")
            print("-" * 50)

            for i, result in enumerate(results["results"], 1):
                print(f"\nResult {i}:")
                print(f"ID: {result['_id']}")
                print(f"Score: {result['score']:.4f}")
                print(f"Username: {result['username']}")
                print(f"User ID: {result['user_id']}")
                print(f"Contact Details: {result['contact_details']}")
                print(f"Total Experience: {result['total_experience']} years")
                print(f"Notice Period: {result['notice_period']} days")
                print(
                    f"Current Salary: {result['current_salary']} {result['currency']} ({result['pay_duration']})"
                )
                print(
                    f"Expected Salary: {result['expected_salary']} {result['currency']} ({result['pay_duration']})"
                )
                print(f"Hike: {result['hike']}%")
                print(f"Skills: {', '.join(result['skills'][:5])}...")
                print(
                    f"Additional Skills: {', '.join(result['may_also_known_skills'][:3])}..."
                )
                print(f"Labels: {', '.join(result['labels'][:3])}...")
                print(f"Last Working Day: {result['last_working_day']}")
                print(f"Tier 1 MBA: {'Yes' if result['is_tier1_mba'] else 'No'}")
                print(
                    f"Tier 1 Engineering: {'Yes' if result['is_tier1_engineering'] else 'No'}"
                )
                print(f"Source: {result['source']}")
                print("\nAcademic Details:")
                for edu in result["academic_details"]:
                    print(f"  - {edu}")
                print("\nExperience:")
                for exp in result["experience"][:2]:  # Show first 2 experiences
                    print(f"  - {exp}")
                print("-" * 50)

    except Exception as e:
        print(f"💥 An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
