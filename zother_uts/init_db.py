from pymongo import MongoClient
from core.custom_logger import CustomLogger
import sys
import os

# Add project root to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from properties.mango import MONGODB_URI, DB_NAME, COLLECTION_NAME

logger = CustomLogger().get_logger("init_db")


def initialize_database():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)

        # Create database and collection
        db = client[DB_NAME]
        if COLLECTION_NAME not in db.list_collection_names():
            # Create collection with a dummy document (required by Atlas)
            db.create_collection(COLLECTION_NAME)
            db[COLLECTION_NAME].insert_one({"_id": "dummy", "initialization": True})
            logger.info(f"Created collection {COLLECTION_NAME} in database {DB_NAME}")

        # Now create the search index
        command = {
            "createSearchIndexes": COLLECTION_NAME,
            "indexes": [
                {
                    "name": "vector_search_index",
                    "definition": {
                        "mappings": {
                            "dynamic": False,
                            "fields": {
                                "skills_vector": {
                                    "type": "knnVector",
                                    "dimensions": 384,
                                    "similarity": "cosine",
                                },
                                "experience_text_vector": {
                                    "type": "knnVector",
                                    "dimensions": 384,
                                    "similarity": "cosine",
                                },
                                "education_text_vector": {
                                    "type": "knnVector",
                                    "dimensions": 384,
                                    "similarity": "cosine",
                                },
                                "combined_resume_vector": {
                                    "type": "knnVector",
                                    "dimensions": 384,
                                    "similarity": "cosine",
                                },
                            },
                        }
                    },
                }
            ],
        }

        result = db.command(command)
        logger.info(f"Search index created successfully: {result}")

        # Remove dummy document if it exists
        db[COLLECTION_NAME].delete_one({"_id": "dummy"})

        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return False


if __name__ == "__main__":
    initialize_database()
