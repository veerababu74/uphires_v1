# File: resume_api/config.py
"""
DEPRECATED: This file is deprecated and will be removed in future versions.
Please use core.config.AppConfig directly instead.

This file exists for backward compatibility only.
"""
import warnings
from core.config import AppConfig

# Issue deprecation warning
warnings.warn(
    "properties.mango is deprecated. Use core.config.AppConfig directly.",
    DeprecationWarning,
    stacklevel=2,
)

# Use configuration from AppConfig
MONGODB_URI = AppConfig.MONGODB_URI
DB_NAME = AppConfig.DB_NAME
COLLECTION_NAME = AppConfig.COLLECTION_NAME
MODEL_NAME = AppConfig.MODEL_NAME
DIMENSIONS = AppConfig.DIMENSIONS
