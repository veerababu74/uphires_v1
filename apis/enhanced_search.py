from fastapi import APIRouter, HTTPException, Query, File, UploadFile
from typing import List, Dict, Any, Optional
from Rag.runner import initialize_rag_app, ask_resume_question_enhanced
from core.custom_logger import CustomLogger
import os
from pathlib import Path

# Initialize logger
logger = CustomLogger().get_logger("enhanced_search")

# Create router instance
router = APIRouter(
    prefix="/enhanced-search",
    tags=["Enhanced Search"],
    responses={404: {"description": "Not found"}},
)

# Configuration for performance presets
PERFORMANCE_PRESETS = {
    "fast": {"mongodb_limit": 20, "llm_limit": 3},
    "balanced": {"mongodb_limit": 50, "llm_limit": 10},
    "comprehensive": {"mongodb_limit": 100, "llm_limit": 20},
    "exhaustive": {"mongodb_limit": 200, "llm_limit": 30},
}


@router.post(
    "/smart-search",
    response_model=Dict[str, Any],
    summary="Smart Search with RAG and Vector Similarity",
    description="""
    Perform a smart search that combines RAG and vector similarity capabilities.
    
    **Parameters:**
    - query: The search query text
    - preset: Performance preset ("fast", "balanced", "comprehensive", "exhaustive")
    - min_score: Minimum similarity score (0.0 to 1.0)
    
    **Returns:**
    Dictionary containing:
    - total_found: Total number of matches found
    - statistics: Search statistics
    - results: List of matching resumes with relevance scores
    """,
)
async def smart_search(
    query: str = Query(..., description="Search query text"),
    preset: str = Query(
        default="balanced",
        description="Performance preset",
        enum=list(PERFORMANCE_PRESETS.keys()),
    ),
    min_score: float = Query(
        default=0.0,
        description="Minimum similarity score",
        ge=0.0,
        le=1.0,
    ),
):
    """
    Perform smart search combining RAG and vector similarity.
    """
    try:
        # Initialize RAG application
        rag_app = initialize_rag_app()

        # Get preset configuration
        config = PERFORMANCE_PRESETS[preset]

        # Perform enhanced RAG search
        rag_result = ask_resume_question_enhanced(
            query,
            mongodb_limit=config["mongodb_limit"],
            llm_limit=config["llm_limit"],
        )

        if "error" in rag_result:
            raise HTTPException(status_code=500, detail=rag_result["error"])

        # Filter results by minimum score
        if "scored_documents" in rag_result:
            rag_result["scored_documents"] = [
                doc
                for doc in rag_result["scored_documents"]
                if doc.get("score", 0) >= min_score
            ]

        return rag_result

    except Exception as e:
        logger.error(f"Smart search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/context-search",
    response_model=Dict[str, Any],
    summary="Context-Aware Resume Search",
    description="""
    Perform a context-aware search that uses LLM to understand and match resume content.
    
    **Parameters:**
    - query: The search query text
    - context_size: Number of documents to analyze (1-20)
    - min_relevance: Minimum relevance score (0.0 to 1.0)
    
    **Returns:**
    Dictionary containing:
    - total_found: Total number of matches found
    - total_analyzed: Number of documents analyzed
    - statistics: Search statistics
    - results: List of matching resumes with relevance scores
    """,
)
async def context_search(
    query: str = Query(..., description="Search query text"),
    context_size: int = Query(
        default=5,
        description="Number of documents to analyze",
        ge=1,
        le=20,
    ),
    min_relevance: float = Query(
        default=0.0,
        description="Minimum relevance score",
        ge=0.0,
        le=1.0,
    ),
):
    """
    Perform context-aware search using LLM.
    """
    try:
        # Initialize RAG application
        rag_app = initialize_rag_app()

        # Perform LLM context search
        result = rag_app.llm_context_search(query, context_size)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Filter results by minimum relevance
        if "results" in result:
            result["results"] = [
                res
                for res in result["results"]
                if res.get("relevance_score", 0) >= min_relevance
            ]

        return result

    except Exception as e:
        logger.error(f"Context search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/jd-based-search",
    response_model=Dict[str, Any],
    summary="Job Description Based Resume Search",
    description="""
    Upload a job description file and find matching resumes using advanced search capabilities.
    
    **Parameters:**
    - file: Job description file (.txt, .pdf, or .docx)
    - preset: Performance preset ("fast", "balanced", "comprehensive", "exhaustive")
    - min_score: Minimum similarity score (0.0 to 1.0)
    
    **Returns:**
    Dictionary containing:
    - total_found: Total number of matches found
    - statistics: Search statistics
    - results: List of matching resumes with relevance scores
    """,
)
async def jd_based_search(
    file: UploadFile = File(...),
    preset: str = Query(
        default="balanced",
        description="Performance preset",
        enum=list(PERFORMANCE_PRESETS.keys()),
    ),
    min_score: float = Query(
        default=0.0,
        description="Minimum similarity score",
        ge=0.0,
        le=1.0,
    ),
):
    """
    Perform search based on job description file.
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        # Save uploaded file
        file_location = temp_dir / file.filename
        with open(file_location, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        try:
            # Extract text from file
            _, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in [".txt", ".pdf", ".docx"]:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type. Only .txt, .pdf, and .docx are supported.",
                )

            # TODO: Implement text extraction from different file types
            # For now, just read text files
            if file_extension.lower() == ".txt":
                with open(file_location, "r", encoding="utf-8") as f:
                    jd_text = f.read()
            else:
                raise HTTPException(
                    status_code=400,
                    detail="PDF and DOCX support coming soon. Please use TXT files for now.",
                )

            if not jd_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Extracted job description is empty.",
                )

            # Get preset configuration
            config = PERFORMANCE_PRESETS[preset]

            # Initialize RAG application
            rag_app = initialize_rag_app()

            # Perform enhanced RAG search with JD text
            result = ask_resume_question_enhanced(
                jd_text,
                mongodb_limit=config["mongodb_limit"],
                llm_limit=config["llm_limit"],
            )

            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            # Filter results by minimum score
            if "scored_documents" in result:
                result["scored_documents"] = [
                    doc
                    for doc in result["scored_documents"]
                    if doc.get("score", 0) >= min_score
                ]

            return result

        finally:
            # Clean up the temporary file
            try:
                os.remove(file_location)
                logger.info(f"Deleted temporary file: {file_location}")
            except Exception as e:
                logger.error(f"Failed to delete temporary file {file_location}: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JD-based search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
