# resume_api/indexes/search_indexes.py
from fastapi import HTTPException
from core.custom_logger import CustomLogger

# Initialize logger
logger = CustomLogger().get_logger("search_indexes")


def check_atlas_search_availability(collection):
    """Check if Atlas Search is available for the collection"""
    try:
        collection.database.command({"listSearchIndexes": collection.name})
        return True
    except Exception as e:
        if "CommandNotFound" in str(e):
            logger.error("Atlas Search is not enabled. Please ensure you are:")
            logger.error("1. Using MongoDB Atlas (not standalone MongoDB)")
            logger.error("2. Have enabled Atlas Search in your cluster settings")
            logger.error("3. Have appropriate permissions")
            return False
        raise e


def create_vector_search_index(collection):
    try:
        # First check if Atlas Search is available
        if not check_atlas_search_availability(collection):
            return False

        # First check if index exists
        existing_indexes = (
            collection.database.command({"listSearchIndexes": collection.name})
            .get("cursor", {})
            .get("firstBatch", [])
        )

        # Check if index already exists and is correctly configured
        for index in existing_indexes:
            if index.get("name") == "vector_search_index":
                # Verify the index definition matches what we need
                existing_def = index.get("definition", {}).get("mappings", {})
                required_fields = [
                    "total_resume_vector",
                    "skills_vector",
                    "experience_text_vector",
                    "education_text_vector",
                    "projects_text_vector",
                    "total_resume_vector",
                ]

                # Check if all required fields exist with correct configuration
                fields = existing_def.get("fields", {})
                if all(
                    field in fields
                    and fields[field].get("type") == "knnVector"
                    and fields[field].get("dimensions") == 384
                    for field in required_fields
                ):
                    logger.info(
                        "Vector search index already exists with correct configuration"
                    )
                    return True
                else:
                    logger.info(
                        "Existing index found but with incorrect configuration. Recreating..."
                    )
                    # Drop the existing index
                    collection.database.command(
                        {
                            "dropSearchIndex": index.get("name"),
                            "collection": collection.name,
                        }
                    )
                    break

        # Create new index with updated definition
        index_definition = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "total_resume_vector": {
                        "type": "knnVector",
                        "dimensions": 384,
                        "similarity": "cosine",
                    },
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
                    "projects_text_vector": {
                        "type": "knnVector",
                        "dimensions": 384,
                        "similarity": "cosine",
                    },
                    "total_resume_vector": {
                        "type": "knnVector",
                        "dimensions": 384,
                        "similarity": "cosine",
                    },
                },
            }
        }

        # Create the index only if needed
        result = collection.database.command(
            {
                "createSearchIndexes": collection.name,
                "indexes": [
                    {"name": "vector_search_index", "definition": index_definition}
                ],
            }
        )

        logger.info(f"Vector search index created successfully: {result}")
        return True

    except Exception as e:
        logger.error(f"Error managing vector search index: {str(e)}, full error: {e}")
        return False


def verify_vector_search_index(collection):
    try:
        # First check if Atlas Search is available
        if not check_atlas_search_availability(collection):
            return False

        indexes = (
            collection.database.command({"listSearchIndexes": collection.name})
            .get("cursor", {})
            .get("firstBatch", [])
        )

        for index in indexes:
            if index.get("name") == "vector_search_index":
                logger.info("Vector search index exists and seems properly configured")
                return True

        logger.info("Vector search index not found!")
        return False
    except Exception as e:
        logger.error(f"Error verifying index: {str(e)}, full error: {e}")
        return False
