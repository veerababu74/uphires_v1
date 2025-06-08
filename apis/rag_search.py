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
from core.custom_logger import CustomLogger
from recent_search_uts.recent_ai_search import save_ai_search_to_recent

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


class SearchError(Exception):
    """Custom exception for search-related errors"""

    pass


# Pydantic models for request bodies
class VectorSimilaritySearchRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User ID who performed the search")
    query: str = Field(..., description="Search query text")
    limit: int = Field(
        default=50, description="Maximum number of results to return", ge=1, le=100
    )


class LLMContextSearchRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User ID who performed the search")
    query: str = Field(..., description="Search query text")
    context_size: int = Field(
        default=5, description="Number of documents to analyze", ge=1, le=20
    )


# Pydantic models for response bodies
class ContactDetails(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    alternative_phone: str = ""
    current_city: str = ""
    looking_for_jobs_in: List[str] = []
    gender: str = ""
    age: int = 0
    naukri_profile: str = ""
    linkedin_profile: str = ""
    portfolio_link: str = ""
    pan_card: str = ""
    aadhar_card: str = ""


class VectorSearchResult(BaseModel):
    _id: str
    contact_details: ContactDetails
    total_experience: str = ""
    notice_period: str = ""
    currency: str = ""
    pay_duration: str = ""
    current_salary: float = 0.0
    hike: float = 0.0
    expected_salary: float = 0.0
    skills: List[str] = []
    may_also_known_skills: List[str] = []
    labels: List[str] = []
    experience: List[Dict[str, Any]] = []
    academic_details: List[Dict[str, Any]] = []
    source: str = ""
    last_working_day: str = ""
    is_tier1_mba: bool = False
    is_tier1_engineering: bool = False
    comment: str = ""
    exit_reason: str = ""
    similarity_score: float


class LLMSearchResult(BaseModel):
    _id: str
    contact_details: ContactDetails
    total_experience: str = ""
    notice_period: str = ""
    currency: str = ""
    pay_duration: str = ""
    current_salary: float = 0.0
    hike: float = 0.0
    expected_salary: float = 0.0
    skills: List[str] = []
    may_also_known_skills: List[str] = []
    labels: List[str] = []
    experience: List[Dict[str, Any]] = []
    academic_details: List[Dict[str, Any]] = []
    source: str = ""
    last_working_day: str = ""
    is_tier1_mba: bool = False
    is_tier1_engineering: bool = False
    comment: str = ""
    exit_reason: str = ""
    relevance_score: float
    match_reason: str = ""


class SearchStatistics(BaseModel):
    retrieved: int = 0
    analyzed: Optional[int] = None
    query: str = ""


class VectorSimilaritySearchResponse(BaseModel):
    results: List[VectorSearchResult]
    total_found: int
    statistics: SearchStatistics


class LLMContextSearchResponse(BaseModel):
    results: List[LLMSearchResult]
    total_analyzed: int
    statistics: SearchStatistics


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
    tags=["ai rag search"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/vector-similarity-search",
    response_model=VectorSimilaritySearchResponse,
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
        # Save the search to recent searches if user_id is provided
        if request.user_id:
            await save_ai_search_to_recent(request.user_id, request.query)
        return result

    except Exception as e:
        logger.error(f"Vector similarity search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/llm-context-search",
    response_model=LLMContextSearchResponse,
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

        # save the search to recent searches
        if request.user_id:
            await save_ai_search_to_recent(request.user_id, request.query)

        return result

    except Exception as e:
        logger.error(f"LLM context search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/llm-context-search/by-jd",
    response_model=LLMContextSearchResponse,
    summary="AI-Powered Resume Search Based on Job Description File",
    description="""
    Upload a job description file (.txt, .pdf, or .docx) and find matching resumes.
    
    The system will extract and clean the text from the file and use AI to find semantically relevant candidates.
    """,
)
async def llm_search_by_jd(
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

            # Extract text and clean it
            jd_text = extract_and_clean_text(file_location)

            # Additional cleaning to remove code-like content
            if jd_text.startswith("#") or "import " in jd_text or "def " in jd_text:
                # This appears to be code, not a job description
                raise SearchError(
                    "Invalid job description format. The file appears to contain code instead of a job description."
                )

            if not jd_text.strip():
                raise SearchError("Extracted job description is empty.")

            # Log the cleaned text for debugging
            logger.info(
                f"Cleaned job description text: {jd_text[:200]}..."
            )  # Log first 200 chars

        finally:
            # Clean up the temporary file
            try:
                os.remove(file_location)
                logging.info(f"Deleted temporary file: {file_location}")
            except Exception as e:
                logging.error(f"Failed to delete temporary file {file_location}: {e}")

        try:
            # Initialize RAG application
            logger.info(
                f"Initializing RAG application for JD search with limit: {limit}"
            )
            rag_app = initialize_rag_app()

            # Perform LLM context search
            logger.info(
                f"Performing LLM context search with text length: {len(jd_text)}"
            )
            result = rag_app.llm_context_search(jd_text, context_size=limit)

            # Log the result for debugging
            logger.info(f"LLM context search completed. Result type: {type(result)}")
            if isinstance(result, dict):
                logger.info(
                    f"Result keys: {list(result.keys()) if result else 'Empty dict'}"
                )

            # Check for errors in result
            if "error" in result:
                logger.error(f"RAG application returned error: {result['error']}")
                raise HTTPException(status_code=500, detail=result["error"])

            # Validate result structure
            if not isinstance(result, dict):
                logger.error(f"Invalid result type: {type(result)}")
                raise HTTPException(
                    status_code=500, detail="Invalid response format from RAG system"
                )

            # Format results to match the expected structure
            formatted_results = []
            for candidate in result.get("results", []):
                formatted_candidate = {
                    "contact_details": {
                        "name": candidate.get("contact_details", {}).get("name", ""),
                        "email": candidate.get("contact_details", {}).get("email", ""),
                        "phone": candidate.get("contact_details", {}).get("phone", ""),
                        "alternative_phone": candidate.get("contact_details", {}).get(
                            "alternative_phone", ""
                        ),
                        "current_city": candidate.get("contact_details", {}).get(
                            "current_city", ""
                        ),
                        "looking_for_jobs_in": candidate.get("contact_details", {}).get(
                            "looking_for_jobs_in", []
                        ),
                        "gender": candidate.get("contact_details", {}).get(
                            "gender", ""
                        ),
                        "age": candidate.get("contact_details", {}).get("age", 0),
                        "naukri_profile": candidate.get("contact_details", {}).get(
                            "naukri_profile", ""
                        ),
                        "linkedin_profile": candidate.get("contact_details", {}).get(
                            "linkedin_profile", ""
                        ),
                        "portfolio_link": candidate.get("contact_details", {}).get(
                            "portfolio_link", ""
                        ),
                        "pan_card": candidate.get("contact_details", {}).get(
                            "pan_card", ""
                        ),
                        "aadhar_card": candidate.get("contact_details", {}).get(
                            "aadhar_card", ""
                        ),
                    },
                    "total_experience": str(candidate.get("total_experience", "0.0")),
                    "notice_period": candidate.get("notice_period", ""),
                    "currency": candidate.get("currency", ""),
                    "pay_duration": candidate.get("pay_duration", ""),
                    "current_salary": float(candidate.get("current_salary", 0)),
                    "hike": float(candidate.get("hike", 0)),
                    "expected_salary": float(candidate.get("expected_salary", 0)),
                    "skills": candidate.get("skills", []),
                    "may_also_known_skills": candidate.get("may_also_known_skills", []),
                    "labels": candidate.get("labels", []),
                    "experience": candidate.get("experience", []),
                    "academic_details": candidate.get("academic_details", []),
                    "source": candidate.get("source", ""),
                    "last_working_day": candidate.get("last_working_day", ""),
                    "is_tier1_mba": bool(candidate.get("is_tier1_mba", False)),
                    "is_tier1_engineering": bool(
                        candidate.get("is_tier1_engineering", False)
                    ),
                    "comment": candidate.get("comment", ""),
                    "exit_reason": candidate.get("exit_reason", ""),
                    "relevance_score": float(candidate.get("relevance_score", 0.0)),
                    "match_reason": candidate.get("match_reason", ""),
                }
                formatted_results.append(formatted_candidate)

            # Format response according to LLMContextSearchResponse model
            formatted_response = {
                "results": formatted_results,
                "total_analyzed": result.get("total_analyzed", 0),
                "statistics": {
                    "retrieved": result.get("statistics", {}).get("retrieved", 0),
                    "analyzed": result.get("statistics", {}).get("analyzed", 0),
                    "query": (
                        jd_text[:200] + "..." if len(jd_text) > 200 else jd_text
                    ),  # Truncate long queries
                },
            }

            logger.info("LLM context search by JD completed successfully")
            return formatted_response

        except Exception as e:
            logger.error(f"Error during RAG search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"RAG search failed: {str(e)}")

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.post(
    "/vector-similarity-search/by-jd",
    response_model=VectorSimilaritySearchResponse,
    summary="AI-Powered Resume Search Based on Job Description File",
    description="""
    Upload a job description file (.txt, .pdf, or .docx) and find matching resumes.
    
    The system will extract and clean the text from the file and use AI to find semantically relevant candidates.
    """,
)
async def vector_search_by_jd(
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
                logging.error(
                    f"Failed to delete temporary file {file_location}: {e}"
                )  # Step 3: Generate embedding from cleaned JD text
        try:
            # Initialize RAG application
            rag_app = initialize_rag_app()

            # Perform vector similarity search
            result = rag_app.vector_similarity_search(jd_text, limit)
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
