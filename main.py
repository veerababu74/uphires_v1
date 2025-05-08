# resume_api/main.py
from fastapi import FastAPI
from api import (
    autocomplete_job_title_skills,
    autocomplete_city,
    manual_resume_search,
    manual,
    ai_resume_search,
    skills_experince_title,
)
from fastapi.middleware.cors import CORSMiddleware
from masking.routes import router as masking_router
from GroqcloudLLM.routes import router as groqcloud_router
from indexes.search_indexes import (
    create_vector_search_index,
    verify_vector_search_index,
)
from database.client import get_collection, get_skills_titles_collection
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("main")

app = FastAPI(
    title="Resume API", description="API for managing resume data with vector search"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize database collection
collection = get_collection()
skills_titles_collection = get_skills_titles_collection()
# Initialize logger

# Create vector search index
create_vector_search_index(collection)

# Include routers

app.include_router(autocomplete_job_title_skills.router)
app.include_router(autocomplete_city.router)
app.include_router(manual_resume_search.router)
app.include_router(skills_experince_title.router)
# app.include_router(manual.router)
app.include_router(ai_resume_search.enhanced_search_router)
app.include_router(masking_router, prefix="/masking", tags=["Masking"])
app.include_router(groqcloud_router, prefix="/groqcloud", tags=["GroqcloudLLM"])


@app.on_event("startup")
async def startup_event():
    # Create regular indexes
    collection.create_index("name")
    collection.create_index("skills")
    collection.create_index("education.institution")
    collection.create_index("education.degree")
    collection.create_index("experience.company")
    collection.create_index("experience.title")
    collection.create_index("total_experience")
    collection.create_index("projects.name")
    collection.create_index("projects.technologies")
    collection.create_index("projects.role")
    collection.create_index("contact_details.address")

    # Create text index for full-text search
    collection.create_index(
        [
            ("name", "text"),
            ("skills", "text"),
            ("education.institution", "text"),
            ("education.degree", "text"),
            ("experience.company", "text"),
            ("experience.title", "text"),
            ("projects.name", "text"),
            ("projects.description", "text"),
        ]
    )

    # Test database connection
    try:
        collection.database.client.admin.command("ping")
        logger.info("Connected to MongoDB successfully!")
        create_vector_search_index(collection)
        if not verify_vector_search_index(collection):
            raise Exception("Index verification failed")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000, reload=True)
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
