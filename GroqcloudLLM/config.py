# GroqCloud LLM Configuration
"""
Configuration settings for GroqCloud LLM integration
Integrates with core configuration system
"""

import os
from core.config import config

# Model Settings - Use core configuration
GROQ_PRIMARY_MODEL = os.getenv("GROQ_PRIMARY_MODEL", "gemma2-9b-it")
GROQ_BACKUP_MODEL = os.getenv("GROQ_BACKUP_MODEL", "llama-3.1-70b-versatile")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "mixtral-8x7b-32768")

# Generation Parameters
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
GROQ_TOP_P = float(os.getenv("GROQ_TOP_P", "0.8"))

# Performance Settings
GROQ_REQUEST_TIMEOUT = int(os.getenv("GROQ_REQUEST_TIMEOUT", "60"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
GROQ_RETRY_DELAY = float(os.getenv("GROQ_RETRY_DELAY", "1.0"))

# Rate Limiting
GROQ_REQUESTS_PER_MINUTE = int(os.getenv("GROQ_REQUESTS_PER_MINUTE", "30"))
GROQ_TOKENS_PER_MINUTE = int(os.getenv("GROQ_TOKENS_PER_MINUTE", "6000"))

# Processing Settings
GROQ_MAX_CONTEXT_LENGTH = int(os.getenv("GROQ_MAX_CONTEXT_LENGTH", "8000"))
MAX_RESUME_LENGTH = 5000  # Truncate long resumes
ENABLE_FALLBACK = True  # Use regex fallback if JSON fails
ENABLE_DEBUG = os.getenv("ENABLE_DEBUG_LOGGING", "false").lower() == "true"

# Legacy compatibility
TEMPERATURE = GROQ_TEMPERATURE
DEFAULT_MODEL = GROQ_PRIMARY_MODEL


def get_groq_config():
    """Get GroqCloud configuration dictionary"""
    return {
        "primary_model": GROQ_PRIMARY_MODEL,
        "backup_model": GROQ_BACKUP_MODEL,
        "fallback_model": GROQ_FALLBACK_MODEL,
        "temperature": GROQ_TEMPERATURE,
        "max_tokens": GROQ_MAX_TOKENS,
        "top_p": GROQ_TOP_P,
        "request_timeout": GROQ_REQUEST_TIMEOUT,
        "max_retries": GROQ_MAX_RETRIES,
        "retry_delay": GROQ_RETRY_DELAY,
        "requests_per_minute": GROQ_REQUESTS_PER_MINUTE,
        "tokens_per_minute": GROQ_TOKENS_PER_MINUTE,
        "max_context_length": GROQ_MAX_CONTEXT_LENGTH,
        "enable_debug": ENABLE_DEBUG,
    }
