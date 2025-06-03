from pymongo import MongoClient
from pymongo.collection import Collection
from typing import Optional
import os
from .config import AppConfig

# Global MongoDB client
_client: Optional[MongoClient] = None


def get_database() -> Collection:
    """Get MongoDB collection for resumes"""
    global _client

    if _client is None:
        _client = MongoClient(AppConfig.MONGODB_URI)

    db = _client[AppConfig.DB_NAME]
    return db[AppConfig.COLLECTION_NAME]
