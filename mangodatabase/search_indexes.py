from pymongo import MongoClient
from core.custom_logger import CustomLogger
import sys
import os

# Add project root to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from properties.mango import MONGODB_URI, DB_NAME, COLLECTION_NAME

logger = CustomLogger().get_logger("search_index_manager")


class SearchIndexManager:
    def __init__(self):
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]

    def check_search_index_exists(self, index_name="vector_search_index"):
        """Check if a search index exists"""
        try:
            # List all search indexes
            indexes = list(self.collection.list_search_indexes())
            for index in indexes:
                if index.get("name") == index_name:
                    logger.info(f"Search index '{index_name}' already exists")
                    return True
            logger.info(f"Search index '{index_name}' does not exist")
            return False
        except Exception as e:
            logger.error(f"Error checking search index: {str(e)}")
            return False

    def get_default_search_index_definition(self):
        """Get the default search index definition"""
        return {
            "name": "vector_search_index",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "combined_resume_vector": {
                            "type": "knnVector",
                            "dimensions": 1024,
                            "similarity": "cosine",
                        },
                        "skills_vector": {
                            "type": "knnVector",
                            "dimensions": 1024,
                            "similarity": "cosine",
                        },
                        "experience_text_vector": {
                            "type": "knnVector",
                            "dimensions": 1024,
                            "similarity": "cosine",
                        },
                        "academic_details_vector": {
                            "type": "knnVector",
                            "dimensions": 1024,
                            "similarity": "cosine",
                        },
                    },
                }
            },
        }

    def create_search_index(self, index_definition=None):
        """Create a search index"""
        try:
            if index_definition is None:
                index_definition = self.get_default_search_index_definition()

            command = {
                "createSearchIndexes": COLLECTION_NAME,
                "indexes": [index_definition],
            }

            result = self.db.command(command)
            logger.info(f"Search index created successfully: {result}")
            return True, result
        except Exception as e:
            logger.error(f"Failed to create search index: {str(e)}")
            return False, str(e)

    def delete_search_index(self, index_name):
        """Delete a search index"""
        try:
            command = {"dropSearchIndexes": COLLECTION_NAME, "indexes": [index_name]}
            result = self.db.command(command)
            logger.info(f"Search index '{index_name}' deleted successfully: {result}")
            return True, result
        except Exception as e:
            logger.error(f"Failed to delete search index '{index_name}': {str(e)}")
            return False, str(e)

    def list_search_indexes(self):
        """List all search indexes"""
        try:
            indexes = list(self.collection.list_search_indexes())
            logger.info(f"Found {len(indexes)} search indexes")
            return True, indexes
        except Exception as e:
            logger.error(f"Failed to list search indexes: {str(e)}")
            return False, str(e)

    def update_search_index(self, index_name, new_definition):
        """Update a search index by deleting and recreating it"""
        try:
            # First delete the existing index
            delete_success, delete_result = self.delete_search_index(index_name)
            if not delete_success:
                return False, f"Failed to delete existing index: {delete_result}"

            # Then create the new index
            create_success, create_result = self.create_search_index(new_definition)
            if not create_success:
                return False, f"Failed to create new index: {create_result}"

            logger.info(f"Search index '{index_name}' updated successfully")
            return True, "Index updated successfully"
        except Exception as e:
            logger.error(f"Failed to update search index '{index_name}': {str(e)}")
            return False, str(e)


def initialize_database():
    """Initialize database and create search index if it doesn't exist"""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]

        # Create collection if it doesn't exist
        if COLLECTION_NAME not in db.list_collection_names():
            db.create_collection(COLLECTION_NAME)
            db[COLLECTION_NAME].insert_one({"_id": "dummy", "initialization": True})
            logger.info(f"Created collection {COLLECTION_NAME} in database {DB_NAME}")

            # Remove dummy document
            db[COLLECTION_NAME].delete_one({"_id": "dummy"})

        # Initialize search index manager
        index_manager = SearchIndexManager()

        # Check if search index exists, if not create it
        if not index_manager.check_search_index_exists():
            success, result = index_manager.create_search_index()
            if success:
                logger.info("Search index created during initialization")
            else:
                logger.error(
                    f"Failed to create search index during initialization: {result}"
                )
                return False
        else:
            logger.info("Search index already exists, skipping creation")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return False


if __name__ == "__main__":
    success = initialize_database()
    if success:
        print("Database initialized successfully")
    else:
        print("Database initialization failed")
