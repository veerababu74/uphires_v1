# resume_api/main.py
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

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from apis.add_userdata import router as add_urer_data
from apis.citys import router as citys_router
from apis.manual_search import router as manual_search_router
from apis.autocomplete_skills_titiles import router as autocomplete_router
from apis.skills_experince_titles import router as skills_experience_titles_router
from apis.vector_search import enhanced_search_router as vector_search_router
from apis.vectore_search_v2 import enhanced_search_router as vector_search_v2_router
from apisofmango.search_index_api import router as search_index_router
from apis.rag_search import router as rag_search_router
from apis.retriever_api import router as retriever_api_router
from masking.routes import router as masking_router
from GroqcloudLLM.routes import router as groqcloud_router
from apisofmango.resume import router as resume_router
from apis.resumerpaser import router as resume_parser_router
from apis.add_userdata import router as add_urer_data
from apis.manual import router as manual_search_router_old
from core.custom_logger import CustomLogger
from core.config import config

# from apis.healthcheck import router as health_router

# Import main functions from separate module
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
        search_index_manager, startup_success = await initialize_application_startup()

        # Set the search index manager in the API router
        from apisofmango.search_index_api import set_search_index_manager

        set_search_index_manager(search_index_manager)

    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise e

    yield

    # Shutdown
    handle_application_shutdown()


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

# Include routers
# app.include_router(health_router)
app.include_router(masking_router, prefix="/masking", tags=["Masking"])
# app.include_router(groqcloud_router, prefix="/groqcloud", tags=["GroqcloudLLM"])
app.include_router(add_urer_data)
app.include_router(
    citys_router,
)
app.include_router(
    manual_search_router,
)
app.include_router(
    autocomplete_router,
)
app.include_router(
    skills_experience_titles_router,
)
app.include_router(
    search_index_router,
    prefix="/search_index",
)
app.include_router(
    vector_search_router,
)

app.include_router(rag_search_router)
app.include_router(resume_router)
app.include_router(resume_parser_router)
app.include_router(
    retriever_api_router,
)
app.include_router(add_urer_data)
app.include_router(manual_search_router_old)


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
