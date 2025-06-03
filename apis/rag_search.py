from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi import status
from typing import List, Dict, Any, Optional
from Rag.runner import initialize_rag_app, ask_resume_question_enhanced
from core.custom_logger import CustomLogger
from pydantic import BaseModel, Field
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
from GroqcloudLLM.text_extraction import extract_and_clean_text

# Initialize logger
logger = CustomLogger().get_logger("rag_search")

# Define base folders
BASE_FOLDER = "dummy_data_save"
TEMP_FOLDER = os.path.join(BASE_FOLDER, "temp_text_extract")
TEMP_DIR = Path(os.path.join(BASE_FOLDER, "temp_files"))

# Ensure the directories exist
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="cleanup.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class SearchError(Exception):
    """Custom exception for search-related errors"""

    pass


# Pydantic models for request bodies
class VectorSimilaritySearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    limit: int = Field(
        default=50, description="Maximum number of results to return", ge=1, le=100
    )


class LLMContextSearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    context_size: int = Field(
        default=5, description="Number of documents to analyze", ge=1, le=20
    )


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


# Create router instance
router = APIRouter(
    prefix="/rag",
    tags=["RAG Search"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/vector-similarity-search",
    response_model=Dict[str, Any],
    summary="Vector Similarity Search",
    description="""
    Perform vector similarity search on resumes using the RAG system.
    
    **Parameters:**
    - query: The search query text
    - limit: Maximum number of results to return (default: 50)
    
    **Returns:**
    Dictionary containing:
    - total_found: Total number of matches found
    - results: List of matching resumes with similarity scores
    """,
    responses={
        200: {
            "description": "Successful search results",
            "content": {
                "application/json": {
                    "example": {
                        "total_found": 10,
                        "results": [
                            {
                                "_id": "resume123",
                                "contact_details": {
                                    "name": "John Doe",
                                    "current_city": "Mumbai",
                                },
                                "skills": ["Python", "React", "AWS"],
                                "total_experience": "5 years",
                                "similarity_score": 0.85,
                            }
                        ],
                    }
                }
            },
        },
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def vector_similarity_search(request: VectorSimilaritySearchRequest):
    """
    Perform vector similarity search on resumes.
    """
    try:
        # Initialize RAG application
        rag_app = initialize_rag_app()

        # Perform vector similarity search
        result = rag_app.vector_similarity_search(request.query, request.limit)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Vector similarity search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/llm-context-search",
    response_model=Dict[str, Any],
    summary="LLM Context Search",
    description="""
    Perform LLM-powered context search on resumes using the RAG system.
    
    **Parameters:**
    - query: The search query text
    - context_size: Number of documents to analyze (default: 5)
    
    **Returns:**
    Dictionary containing:
    - total_found: Total number of matches found
    - total_analyzed: Number of documents analyzed
    - statistics: Search statistics
    - results: List of matching resumes with relevance scores and match reasons
    """,
    responses={
        200: {
            "description": "Successful search results",
            "content": {
                "application/json": {
                    "example": {
                        "total_found": 10,
                        "total_analyzed": 5,
                        "statistics": {
                            "avg_relevance": 0.85,
                            "match_distribution": {"high": 3, "medium": 2, "low": 0},
                        },
                        "results": [
                            {
                                "_id": "resume123",
                                "contact_details": {
                                    "name": "John Doe",
                                    "current_city": "Mumbai",
                                },
                                "skills": ["Python", "React", "AWS"],
                                "total_experience": "5 years",
                                "relevance_score": 0.92,
                                "match_reason": "Strong match in Python and AWS skills",
                            }
                        ],
                    }
                }
            },
        },
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def llm_context_search(request: LLMContextSearchRequest):
    """
    Perform LLM-powered context search on resumes.
    """
    try:
        # Initialize RAG application
        rag_app = initialize_rag_app()

        # Perform LLM context search
        result = rag_app.llm_context_search(request.query, request.context_size)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"LLM context search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/llm-context-search/by-jd",
    response_model=List[Dict[str, Any]],
    summary="AI-Powered Resume Search Based on Job Description File",
    description="""
    Upload a job description file (.txt, .pdf, or .docx) and find matching resumes.
    
    The system will extract and clean the text from the file and use AI to find semantically relevant candidates.
    """,
)
async def search_by_jd(
    file: UploadFile = File(...),
    limit: int = 10,
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
            # Initialize RAG application
            rag_app = initialize_rag_app()

            # Perform LLM context search
            result = rag_app.llm_context_search(jd_text, context_size=limit)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.post(
    "/vector-similarity-search/by-jd",
    response_model=List[Dict[str, Any]],
    summary="AI-Powered Resume Search Based on Job Description File",
    description="""
    Upload a job description file (.txt, .pdf, or .docx) and find matching resumes.
    
    The system will extract and clean the text from the file and use AI to find semantically relevant candidates.
    """,
)
async def search_by_jd(
    file: UploadFile = File(...),
    limit: int = 10,
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
            # Initialize RAG application
            rag_app = initialize_rag_app()

            # Perform LLM context search
            result = rag_app.vector_similarity_search(jd_text, context_size=limit)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
