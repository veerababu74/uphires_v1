# resume_api/main.py
"""
Resume Search API - Main Application Entry Point

This is a FastAPI application for managing resumes with vector embeddings,
AI-powered search, and comprehensive resume parsing capabilities.

Author: Uphire Team
Version: 1.0.0
"""

import warnings

# Suppress cryptography deprecation warnings from PyMongo SSL certificates
try:
    from cryptography.utils import CryptographyDeprecationWarning

    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="pymongo.pyopenssl_context"
)

# Fix Windows Unicode issues early
import sys

if sys.platform.startswith("win"):
    try:
        from fix_windows_unicode import fix_windows_unicode

        fix_windows_unicode()
    except ImportError:
        # Fallback manual fix
        import os

        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import API routers
from apis.add_userdata import router as add_user_data_router
from apis.citys import router as citys_router
from apis.manual_search import router as manual_search_router
from apis.manual_recent_search_save import router as manual_search_save_recent_router
from apis.autocomplete_skills_titiles import router as autocomplete_router
from apis.skills_experince_titles import router as skills_experience_titles_router
from apis.vector_search import enhanced_search_router as vector_search_router
from apis.vectore_search_v2 import enhanced_search_router as vector_search_v2_router
from apisofmango.search_index_api import router as search_index_router
from apis.rag_search import router as rag_search_router
from apis.retriever_api import router as retriever_api_router
from masking.routes import router as masking_router
from GroqcloudLLM.routes import router as groqcloud_router
from multipleresumepraser.routes import router as multiple_resume_parser_router
from apisofmango.resume import router as resume_router
from apis.ollama_test import router as ollama_test_router
from apis.manual import router as manual_search_router_old
from apis.ai_recent_saved_searchs import router as ai_search_save_recent_router
from apis.llm_config_api import router as llm_config_router
from apis.healthcheck import router as health_router
from apis.llm_provider_management import router as llm_provider_router
from apis.resumerpaser import router as resume_parser_router

# from apis.resumerpaser import router as resume_parser_router  # TODO: Fix router definition

# Import core modules
from core.custom_logger import CustomLogger
from core.config import config
from core.auto_model_downloader import ensure_embedding_models_on_startup
from core.production_models import get_deployment_models
from main_functions import initialize_application_startup, handle_application_shutdown

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("main")

# Global search index manager
search_index_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global search_index_manager

    try:
        # Step 1: Ensure embedding models are available
        logger.info("Starting application initialization...")

        # Get production models based on deployment configuration
        # Options: "minimal", "balanced", "full", "complete"
        # Change in core/production_models.py or set via environment
        deployment_type = os.getenv("EMBEDDING_DEPLOYMENT", "balanced")
        production_models = get_deployment_models(deployment_type)

        logger.info(f"Using '{deployment_type}' deployment configuration")
        logger.info(f"Models to ensure: {production_models}")

        model_results = await ensure_embedding_models_on_startup(
            required_models=production_models,
            timeout_seconds=600,  # 10 minutes max for downloads
        )

        # Check if critical models are available
        critical_models = ["BAAI/bge-large-en-v1.5", "BAAI/bge-large-zh-v1.5"]
        missing_critical = [
            model for model in critical_models if not model_results.get(model, False)
        ]

        if missing_critical:
            logger.warning(
                f"Some critical models failed to download: {missing_critical}"
            )
            logger.warning("Application will continue but some features may be limited")
        else:
            logger.info("All critical embedding models are ready!")

        # Step 2: Initialize other application components
        search_index_manager, startup_success = await initialize_application_startup()

        # Set the search index manager in the API router
        from apisofmango.search_index_api import set_search_index_manager

        set_search_index_manager(search_index_manager)

        logger.info("Application startup completed successfully!")

    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise e

    yield

    # Shutdown
    logger.info("🔄 Shutting down application...")
    handle_application_shutdown()
    logger.info("👋 Application shutdown completed")


app = FastAPI(
    title="Resume API",
    description="API for managing resumes with vector embeddings",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers with proper organization and tagging
app.include_router(health_router, prefix="/health", tags=["Health Check"])
app.include_router(masking_router, prefix="/masking", tags=["Masking"])
app.include_router(add_user_data_router, prefix="/api", tags=["User Data"])
app.include_router(citys_router, prefix="/api", tags=["Cities"])
app.include_router(manual_search_router, prefix="/api", tags=["Manual Search"])
app.include_router(
    manual_search_save_recent_router, prefix="/api", tags=["Recent Searches"]
)
app.include_router(autocomplete_router, prefix="/api", tags=["Autocomplete"])
app.include_router(
    skills_experience_titles_router, prefix="/api", tags=["Skills & Experience"]
)
app.include_router(search_index_router, prefix="/search_index", tags=["Search Index"])
app.include_router(vector_search_router, prefix="/api", tags=["Vector Search"])
app.include_router(vector_search_v2_router, prefix="/api/v2", tags=["Vector Search V2"])
app.include_router(rag_search_router, prefix="/api", tags=["RAG Search"])
app.include_router(resume_router, prefix="/api", tags=["Resume Management"])
app.include_router(retriever_api_router, prefix="/api", tags=["Retriever"])
app.include_router(
    manual_search_router_old, prefix="/api/legacy", tags=["Legacy Manual Search"]
)
app.include_router(ai_search_save_recent_router, prefix="/api", tags=["AI Search"])
app.include_router(ollama_test_router, prefix="/api", tags=["Ollama Testing"])
app.include_router(llm_config_router, prefix="/api", tags=["LLM Configuration"])
app.include_router(groqcloud_router, prefix="/groqcloud", tags=["GroqCloudLLM"])
app.include_router(
    multiple_resume_parser_router,
    prefix="/resume_parser",
    tags=["Multiple Resume Parser"],
)
app.include_router(llm_provider_router, prefix="/api", tags=["LLM Provider Management"])
# app.include_router(resume_parser_router, prefix="/resume_parser", tags=["Resume Parser"])  # TODO: Fix router
app.include_router(resume_parser_router, prefix="/api", tags=["Resume Parser"])


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Resume API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.detail, "success": False}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "success": False}
    )


# Health check endpoint to verify vector search status


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000, reload=True)
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
