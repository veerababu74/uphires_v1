import re
import os
import numpy as np
from fastapi import APIRouter, Body, HTTPException, status, UploadFile, File
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from mangodatabase.client import get_collection
from core.helpers import format_resume
from embeddings.vectorizer import Vectorizer
from schemas.vector_search_scehma import VectorSearchQuery
from GroqcloudLLM.text_extraction import extract_and_clean_text
from pathlib import Path

from datetime import datetime, timedelta

BASE_FOLDER = "dummy_data_save"
TEMP_FOLDER = os.path.join(BASE_FOLDER, "temp_text_extract")
TEMP_DIR = Path(os.path.join(BASE_FOLDER, "temp_files"))

# Ensure the directories exist
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
from core.custom_logger import CustomLogger

logger_manager = CustomLogger()
logging = logger_manager.get_logger("vector_search_api")


def cleanup_temp_directory(age_limit_minutes: int = 60):
    """
    Cleanup temporary directory by deleting files older than the specified age limit.
    """
    now = datetime.now()
    for file_path in TEMP_DIR.iterdir():
        if file_path.is_file():
            file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age > timedelta(minutes=age_limit_minutes):
                try:
                    file_path.unlink()
                    logging.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to delete file {file_path}: {e}")


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
enhanced_search_router = APIRouter(prefix="/aiv1", tags=["enhanced ai vector search"])
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
            "education": "academic_details_vector",
            "projects": "experience_text_vector",  # Projects search through experience for better relevance
            "full_text": "combined_resume_vector",
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


@enhanced_search_router.post(
    "/search-by-jd",
    response_model=List[Dict[str, Any]],
    summary="AI-Powered Resume Search Based on Job Description File",
    description="""
    Upload a job description file (.txt, .pdf, or .docx) and find matching resumes.
    
    The system will extract and clean the text from the file and use AI to find semantically relevant candidates.
    """,
)
async def search_by_jd(
    file: UploadFile = File(...),
    field: str = "full_text",
    num_results: int = 10,
    min_score: float = 0.0,
):
    try:
        # Step 1: Save uploaded file to temp directory
        file_location = os.path.join(TEMP_FOLDER, file.filename)

        with open(file_location, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        # Step 2: Extract and clean text from file
        try:
            _, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in [".txt", ".pdf", ".docx"]:
                raise SearchError(
                    "Unsupported file type. Only .txt, .pdf, and .docx are supported."
                )

            jd_text = extract_and_clean_text(file_location)
            if not jd_text.strip():
                raise SearchError("Extracted job description is empty.")
        finally:
            # Clean up the temporary file
            try:
                os.remove(file_location)
                logging.info(f"Deleted temporary file: {file_location}")
            except Exception as e:
                logging.error(f"Failed to delete temporary file {file_location}: {e}")

        # Step 3: Generate embedding from cleaned JD text
        try:
            query_embedding = vectorizer.generate_embedding(jd_text)
        except Exception as e:
            raise SearchError(f"Failed to generate embedding from JD: {str(e)}")

        # Step 4: Map field to vector path
        vector_field_mapping = {
            "skills": "skills_vector",
            "experience": "experience_text_vector",
            "education": "education_text_vector",
            "projects": "projects_text_vector",
            "full_text": "total_resume_vector",
        }

        vector_field = vector_field_mapping.get(field)
        if not vector_field:
            raise SearchError(
                f"Invalid field name. Choose from: {', '.join(vector_field_mapping.keys())}"
            )

        # Step 5: Run vector search pipeline
        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": vector_field,
                        "k": num_results,
                    },
                }
            },
            {"$set": {"score": {"$meta": "searchScore"}}},
            {"$match": {"score": {"$gte": min_score}}},
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
