import re
import numpy as np
from fastapi import APIRouter, Body, HTTPException, status
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from database.client import get_collection
from core.helpers import format_resume
from core.vectorizer import Vectorizer
from models.search import VectorSearchQuery


"""
Vector Search API Documentation:

Field Mapping Options:
- skills: Searches through candidate's technical and soft skills
- experience: Searches through work experience descriptions
- education: Searches through educational background
- projects: Searches through project descriptions
- full_text: Searches across the entire resume

Expected Input Format:
{
    "query": "string",      # Search query text
    "field": "string",      # One of the field mapping options above
    "num_results": int,     # Number of results to return (default: 10)
    "min_score": float      # Minimum similarity score threshold (0.0 to 1.0)
}
"""

# Get MongoDB collection
resumes_collection = get_collection()
enhanced_search_router = APIRouter(prefix="/ai", tags=["enhanced ai vector search"])
vectorizer = Vectorizer()


class SearchError(Exception):
    """Custom exception for search-related errors"""

    pass


@enhanced_search_router.post(
    "/search",
    response_model=List[Dict[str, Any]],
    summary="AI-Powered Resume Search",
    description="""
    Perform semantic search across resume database using AI embeddings.
    
    **Input Fields:**
    - query: Search text (e.g., "Python developer with 5 years experience in machine learning")
    - field: Search scope (default: "full_text")
    - num_results: Number of results to return (default: 10)
    - min_score: Minimum similarity threshold (default: 0.0)
    
    **Example Input:**
    ```json
    {
        "query": "experienced machine learning engineer with python",
        "field": "full_text",
        "num_results": 10,
        "min_score": 0.2
    }
    ```
    
    **Output Fields:**
    - name: Candidate's full name
    - contact_details: Email, phone, location etc.
    - education: List of educational qualifications
    - experience: List of work experiences
    - projects: List of projects
    - total_experience: Years of experience
    - skills: List of technical and soft skills
    - certifications: List of certifications
    - relevance_score: Match score (0-100)
    
    **Search Fields:**
    - full_text: Search entire resume (default)
    - skills: Search only skills section
    - experience: Search work experience
    - education: Search educational background
    - projects: Search project descriptions
    """,
    responses={
        200: {
            "description": "Successful search results",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "name": "John Doe",
                            "contact_details": {
                                "email": "john@example.com",
                                "phone": "+1234567890",
                                "location": "New York, USA",
                            },
                            "education": [
                                {
                                    "degree": "Master of Science in Computer Science",
                                    "institution": "Stanford University",
                                    "year": "2020",
                                }
                            ],
                            "experience": [
                                {
                                    "title": "Senior Machine Learning Engineer",
                                    "company": "Tech Corp",
                                    "duration": "2020-Present",
                                    "description": "Led ML projects using Python and TensorFlow",
                                }
                            ],
                            "projects": [
                                {
                                    "name": "AI Chatbot",
                                    "description": "Developed NLP-based customer service bot",
                                }
                            ],
                            "total_experience": 5.5,
                            "skills": ["Python", "Machine Learning", "TensorFlow"],
                            "certifications": ["AWS Machine Learning Specialty"],
                            "relevance_score": 95.5,
                        }
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search query cannot be empty"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "An unexpected error occurred"}
                }
            },
        },
    },
)
async def vector_search(search_query: VectorSearchQuery):
    try:
        # Input validation
        if not search_query.query.strip():
            raise SearchError("Search query cannot be empty")

        if search_query.num_results < 1:
            raise SearchError("Number of results must be greater than 0")

        # Generate embedding for search query
        try:
            query_embedding = vectorizer.generate_embedding(search_query.query)
        except Exception as e:
            raise SearchError(f"Failed to generate embedding: {str(e)}")

        vector_field_mapping = {
            "skills": "skills_vector",
            "experience": "experience_text_vector",
            "education": "education_text_vector",
            "projects": "projects_text_vector",
            "full_text": "total_resume_vector",
        }

        vector_field = vector_field_mapping.get(search_query.field)
        if not vector_field:
            raise SearchError(
                f"Invalid field name. Choose from: {', '.join(vector_field_mapping.keys())}"
            )

        # Enhanced search pipeline with scoring and filtering
        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": vector_field,
                        "k": search_query.num_results,
                    },
                }
            },
            {"$set": {"score": {"$meta": "searchScore"}}},
            {"$match": {"score": {"$gte": search_query.min_score}}},
            {
                "$project": {
                    "name": 1,
                    "contact_details": 1,
                    "education": 1,
                    "experience": 1,
                    "projects": 1,
                    "total_experience": 1,
                    "skills": 1,
                    "certifications": 1,
                    "score": 1,
                }
            },
            {"$sort": {"score": -1}},
        ]

        try:
            results = list(resumes_collection.aggregate(pipeline))
        except Exception as e:
            raise SearchError(f"Database query failed: {str(e)}")

        if not results:
            return []

        formatted_results = [format_resume(result) for result in results]

        # Add relevance score to results
        for result in formatted_results:
            result["relevance_score"] = round(result.get("score", 0) * 100, 2)

        return formatted_results

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
