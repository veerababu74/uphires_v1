from pymongo import MongoClient
from pymongo.collection import Collection
from typing import Optional
import os

# MongoDB connection settings
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "resume_db")

# Global MongoDB client
_client: Optional[MongoClient] = None


def get_database() -> Collection:
    """Get MongoDB collection for resumes"""
    global _client

    if _client is None:
        _client = MongoClient(MONGODB_URL)

    db = _client[DATABASE_NAME]
    return db["resumes"]
