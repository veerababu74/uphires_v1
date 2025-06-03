import json
from typing import Dict, List, Optional, Tuple
from bson import ObjectId
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo.collection import Collection

from core.custom_logger import CustomLogger
from core.helpers import JSONEncoder
from .config import RAGConfig
from .utils import DocumentProcessor
from .chains import ChainManager

logger = CustomLogger().get_logger("search_engines")


class VectorSearchEngine:
    """Handles pure vector similarity search operations"""

    def __init__(self, vector_store: MongoDBAtlasVectorSearch, collection: Collection):
        self.vector_store = vector_store
        self.collection = collection

    def search(self, query: str, limit: int = 50) -> Dict:
        """Perform pure vector similarity search and return results sorted by score"""
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
                    complete_doc = self.collection.find_one({"_id": ObjectId(doc_id)})

                    if complete_doc:
                        formatted_doc = DocumentProcessor.format_complete_document(
                            complete_doc
                        )
                        formatted_doc["similarity_score"] = float(score)
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


class LLMSearchEngine:
    """Handles LLM-enhanced search operations"""

    def __init__(
        self,
        vector_store: MongoDBAtlasVectorSearch,
        collection: Collection,
        chain_manager: ChainManager,
    ):
        self.vector_store = vector_store
        self.collection = collection
        self.chain_manager = chain_manager
        self.vector_engine = VectorSearchEngine(vector_store, collection)

    def search(self, query: str, context_size: int = 5) -> Dict:
        """Perform LLM-based search with user-controlled context size"""
        try:
            logger.info(f"Performing LLM search for: {query}")
            logger.info(f"Context size: {context_size}")

            # First get documents using vector search
            vector_results = self.vector_engine.search(query, limit=context_size)

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

            # Fetch full documents and format context
            context_string = self._prepare_context(doc_ids)

            # Use LLM to analyze and rank documents
            logger.info("Invoking LLM for document analysis...")
            result = self.chain_manager.ranking_chain.invoke(
                {"context": context_string, "question": query}
            )

            return self._format_llm_results(
                result, query, len(vector_results["results"])
            )

        except Exception as e:
            logger.error(f"Error during LLM search: {e}")
            return {"error": str(e)}

    def _prepare_context(self, doc_ids: List[str]) -> str:
        """Prepare context string from document IDs"""
        projection = {field: 1 for field in RAGConfig.FIELDS_TO_EXTRACT}
        if "_id" not in projection:
            projection["_id"] = 1

        fetched_docs = list(
            self.collection.find(
                {"_id": {"$in": [ObjectId(doc_id) for doc_id in doc_ids]}},
                projection,
            )
        )

        context_parts = []
        for doc in fetched_docs:
            normalized_doc = {}
            for field, value in doc.items():
                if field == "_id":
                    normalized_doc[field] = str(value)
                elif field in ["skills", "may_also_known_skills", "labels"]:
                    normalized_doc[field] = DocumentProcessor.normalize_list_field(
                        value
                    )
                else:
                    normalized_doc[field] = DocumentProcessor.normalize_field_value(
                        value
                    )

            context_parts.append(json.dumps(normalized_doc, indent=2, cls=JSONEncoder))

        return "\n\n---\n\n".join(context_parts)

    def _format_llm_results(
        self, result: Dict, query: str, retrieved_count: int
    ) -> Dict:
        """Format LLM results into standardized format"""
        if isinstance(result, dict) and "matches" in result:
            formatted_results = []
            for match in result["matches"]:
                if isinstance(match, dict) and "_id" in match:
                    complete_doc = self.collection.find_one(
                        {"_id": ObjectId(match["_id"])}
                    )

                    if complete_doc:
                        formatted_doc = DocumentProcessor.format_complete_document(
                            complete_doc
                        )
                        formatted_doc.update(
                            {
                                "relevance_score": float(
                                    match.get("relevance_score", 0.0)
                                ),
                                "match_reason": str(
                                    match.get("match_reason", "No explanation provided")
                                ),
                            }
                        )
                        formatted_results.append(formatted_doc)

            # Sort by relevance score
            formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)

            return {
                "results": formatted_results,
                "total_analyzed": len(formatted_results),
                "statistics": {
                    "retrieved": retrieved_count,
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

    # Additional methods for enhanced functionality...
    def ask_question_with_limits(
        self, question: str, mongodb_limit: int, llm_limit: int
    ) -> Optional[Dict]:
        """Implementation for asking questions with limits"""
        # Implementation here...
        pass

    def get_candidates_with_limits(
        self, question: str, mongodb_limit: int, llm_limit: int, max_results: int
    ) -> Optional[Dict]:
        """Implementation for getting candidates with limits"""
        # Implementation here...
        pass

    def rank_candidates_with_limits(
        self, question: str, mongodb_limit: int, llm_limit: int
    ) -> Optional[Dict]:
        """Implementation for ranking candidates with limits"""
        # Implementation here...
        pass
