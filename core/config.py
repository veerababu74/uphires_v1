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
        "mongodb+srv://veera:Babu7474@uphire-test.gbkcxnd.mongodb.net/?retryWrites=true&w=majority&appName=uphire-test",
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

    # Groq API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEYS", "").split(",")[0].strip()

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
        }


# Global config instance
config = AppConfig()
