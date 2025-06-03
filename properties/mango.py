# File: resume_api/config.py
from core.config import AppConfig

# Use configuration from AppConfig
MONGODB_URI = AppConfig.MONGODB_URI
DB_NAME = AppConfig.DB_NAME
COLLECTION_NAME = AppConfig.COLLECTION_NAME
MODEL_NAME = AppConfig.MODEL_NAME
DIMENSIONS = AppConfig.DIMENSIONS
