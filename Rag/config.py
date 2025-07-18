import os
from dotenv import load_dotenv
from core.config import AppConfig

load_dotenv()


class RAGConfig:
    """Configuration for RAG application

    Note: RAG now uses the centralized LLM configuration system from core.llm_config
    The LLM provider can be switched dynamically using LLMConfigManager.
    The settings below are kept for backward compatibility and fallback scenarios.
    """

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

    # LLM Configuration - Now uses centralized LLM config
    # These are kept for backward compatibility and fallback scenarios
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_PRIMARY_MODEL", "llama3.2:3b")
    OLLAMA_BACKUP_MODEL = os.getenv("OLLAMA_BACKUP_MODEL", "qwen2.5:3b")
    OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "qwen:4b")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "60"))
    OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "60"))

    # Groq Configuration (Legacy - use centralized config instead)
    GROQ_API_KEY = AppConfig.GROQ_API_KEY
    LLM_MODEL = os.getenv("GROQ_PRIMARY_MODEL", "gemma2-9b-it")
    LLM_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.0"))

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
