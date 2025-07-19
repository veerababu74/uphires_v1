# Configuration settings for the application
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AppConfig:
    """Application configuration settings"""

    # MongoDB Configuration
    MONGODB_URI = os.getenv(
        "MONGODB_CONNECTION_STRING",
        "mongodb+srv://veera:Babu7474@uphire-test.aw2gzuy.mongodb.net/?retryWrites=true&w=majority&appName=uphire-test",
    )
    DB_NAME = os.getenv("DB_NAME", "resume_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "resumes")

    # Vector Search Configuration
    VECTOR_FIELD = os.getenv("VECTOR_FIELD", "combined_resume_vector")
    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSIONS = 384

    # Atlas Search Configuration
    ENABLE_ATLAS_SEARCH = os.getenv("ENABLE_ATLAS_SEARCH", "true").lower() == "true"

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Legacy Groq API Configuration (for backward compatibility)
    GROQ_API_KEY = os.getenv("GROQ_API_KEYS", "").split(",")[0].strip()

    # LLM Provider Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

    @classmethod
    def is_atlas_search_enabled(cls) -> bool:
        """Check if Atlas Search features should be enabled"""
        return cls.ENABLE_ATLAS_SEARCH

    @classmethod
    def validate_atlas_configuration(cls) -> bool:
        """Validate that Atlas Search configuration is properly set up"""
        if not cls.ENABLE_ATLAS_SEARCH:
            return False

        if not cls.MONGODB_URI:
            return False

        # Check if connection string is for Atlas (contains 'mongodb.net')
        if "mongodb.net" not in cls.MONGODB_URI:
            return False

        return True

    @classmethod
    def get_connection_info(cls) -> dict:
        """Get connection information for debugging"""
        return {
            "atlas_search_enabled": cls.ENABLE_ATLAS_SEARCH,
            "has_connection_string": bool(cls.MONGODB_URI),
            "is_atlas_connection": bool(
                cls.MONGODB_URI and "mongodb.net" in cls.MONGODB_URI
            ),
            "log_level": cls.LOG_LEVEL,
            "llm_provider": cls.LLM_PROVIDER,
        }

    @classmethod
    def get_llm_info(cls) -> dict:
        """Get LLM configuration information"""
        return {
            "provider": cls.LLM_PROVIDER,
            "groq_api_key_configured": bool(cls.GROQ_API_KEY),
        }


# Global config instance
config = AppConfig()
