"""
Custom exception classes for the Resume Search API

This module defines custom exceptions used throughout the application
for better error handling and logging.
"""

from typing import Optional, Dict, Any


class ResumeAPIException(Exception):
    """Base exception for all Resume API errors"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class DatabaseConnectionError(ResumeAPIException):
    """Raised when database connection fails"""

    pass


class VectorSearchError(ResumeAPIException):
    """Raised when vector search operations fail"""

    pass


class LLMProviderError(ResumeAPIException):
    """Raised when LLM provider operations fail"""

    pass


class ConfigurationError(ResumeAPIException):
    """Raised when configuration is invalid or missing"""

    pass


class ValidationError(ResumeAPIException):
    """Raised when input validation fails"""

    pass


class SearchIndexError(ResumeAPIException):
    """Raised when search index operations fail"""

    pass


class ResumeParsingError(ResumeAPIException):
    """Raised when resume parsing fails"""

    pass


class EmbeddingError(ResumeAPIException):
    """Raised when embedding generation fails"""

    pass
