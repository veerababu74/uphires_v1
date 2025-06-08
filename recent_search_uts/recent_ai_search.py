from mangodatabase.client import get_ai_recent_search_collection
from datetime import datetime, timedelta, timezone
import uuid
from core.custom_logger import CustomLogger

# Initialize logger
logging = CustomLogger().get_logger("recent_ai_search_uts")


async def save_ai_search_to_recent(user_id: str, query: str):
    """Save AI search to recent searches collection"""
    try:
        collection = get_ai_recent_search_collection()

        # Generate unique search ID
        search_id = str(uuid.uuid4())

        # Create search document
        search_document = {
            "search_id": search_id,
            "user_id": user_id,
            "query": query,
            "timestamp": datetime.now(timezone.utc),
            "search_type": "ai",
            "is_saved": True,
        }

        # Check if user has too many recent searches (limit to 100)
        user_recent_count = collection.count_documents({"user_id": user_id})
        if user_recent_count >= 100:
            # Remove oldest search for this user
            oldest_search = collection.find_one(
                {"user_id": user_id}, sort=[("timestamp", 1)]
            )
            if oldest_search:
                collection.delete_one({"_id": oldest_search["_id"]})

        # Insert new recent search
        collection.insert_one(search_document)
        logging.info(f"Saved AI search to recent for user {user_id}: {query}")

    except Exception as e:
        logging.error(f"Failed to save AI search to recent: {str(e)}")
        # Don't raise exception here as this shouldn't break the main search functionality
