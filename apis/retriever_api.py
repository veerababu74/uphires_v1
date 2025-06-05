from fastapi import APIRouter, HTTPException, File, UploadFile, status
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import os
from Retrivers.retriver import MangoRetriever, LangChainRetriever
import logging
from GroqcloudLLM.text_extraction import extract_and_clean_text
from core.config import AppConfig

from datetime import datetime, timedelta
from pathlib import Path

# Initialize router
router = APIRouter(
    prefix="/search",
    tags=["enhanced ai vector search - retriever"],
    responses={404: {"description": "Not found"}},
)

# Initialize retrievers
mango_retriever = MangoRetriever()
langchain_retriever = LangChainRetriever()

# Setup temp directory
TEMP_FOLDER = "temp_uploads"
TEMP_DIR = Path(TEMP_FOLDER)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="retriever_cleanup.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def cleanup_temp_directory(age_limit_minutes: int = 60):
    """Cleanup temporary directory by deleting files older than the specified age limit."""
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


class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5


class SearchResponse(BaseModel):
    results: List[Dict]
    total_count: int
    query: str


class SearchError(Exception):
    """Custom exception for search-related errors"""

    pass


@router.post("/mango", response_model=SearchResponse)
async def search_mango(request: SearchRequest):
    """
    Search using the Mango retriever
    """
    try:
        results = mango_retriever.search_and_rank(request.query, request.limit)
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/langchain", response_model=SearchResponse)
async def search_langchain(request: SearchRequest):
    """
    Search using the LangChain retriever
    """
    try:
        results = langchain_retriever.search_and_rank(request.query, request.limit)
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/mango/search-by-jd",
    response_model=SearchResponse,
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
            results = mango_retriever.search_and_rank(jd_text, limit)

            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.post(
    "/langchain/search-by-jd",
    response_model=SearchResponse,
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
            results = langchain_retriever.search_and_rank(jd_text, limit)

            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
