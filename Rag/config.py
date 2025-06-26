import os
from dotenv import load_dotenv
from core.config import AppConfig

load_dotenv()


class RAGConfig:
    """Configuration for RAG application"""

    # MongoDB Configuration
    MONGODB_URI = AppConfig.MONGODB_URI
    DB_NAME = AppConfig.DB_NAME
    COLLECTION_NAME = AppConfig.COLLECTION_NAME

    # Vector Search Configuration
    MODEL_NAME = AppConfig.MODEL_NAME
    DIMENSIONS = AppConfig.DIMENSIONS

    # Atlas Search Configuration
    ATLAS_SEARCH_INDEX = "resume_search_index"
    ATLAS_SEARCH_ENABLED = True

    # Database Configuration
    VECTOR_FIELD = "combined_resume_vector"
    INDEX_NAME = "vector_search_index"

    # LLM Configuration
    GROQ_API_KEY = AppConfig.GROQ_API_KEY
    LLM_MODEL = "gemma2-9b-it"
    LLM_TEMPERATURE = 0.0

    # ollama Configuration
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    OLLAMA_MODEL = "gemma2-9b-it"
    OLLAMA_TEMPERATURE = 0.0

    # Performance Limits
    MAX_CONTEXT_LENGTH = 8000
    DEFAULT_MONGODB_LIMIT = 50
    DEFAULT_LLM_LIMIT = 10
    DEFAULT_MAX_RESULTS = 20

    # Document Fields
    FIELDS_TO_EXTRACT = [
        "_id",
        "user_id",
        "username",
        "contact_details",
        "total_experience",
        "notice_period",
        "currency",
        "pay_duration",
        "current_salary",
        "hike",
        "expected_salary",
        "skills",
        "may_also_known_skills",
        "labels",
        "experience",
        "academic_details",
        "source",
        "last_working_day",
        "is_tier1_mba",
        "is_tier1_engineering",
    ]

    # Performance Presets
    PERFORMANCE_PRESETS = {
        "fast": {"mongodb_limit": 20, "llm_limit": 3},
        "balanced": {"mongodb_limit": 50, "llm_limit": 10},
        "comprehensive": {"mongodb_limit": 100, "llm_limit": 20},
        "exhaustive": {"mongodb_limit": 200, "llm_limit": 30},
    }
