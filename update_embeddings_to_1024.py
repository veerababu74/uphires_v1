#!/usr/bin/env python3
"""
Script to update embeddings from 384 to 1024 dimensions using BAAI/bge-large-zh-v1.5

This script:
1. Updates MongoDB search index to support 1024 dimensions
2. Re-generates all existing document embeddings with the new model
3. Updates the environment configuration

IMPORTANT:
- Backup your database before running this script
- This operation may take significant time depending on your data size
- Ensure you have sufficient computational resources
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pymongo import MongoClient
from core.custom_logger import CustomLogger
from core.config import AppConfig
from mangodatabase.search_indexes import SearchIndexManager
from embeddings import EmbeddingManager, ResumeVectorizer
from embeddings.manager import AddUserDataVectorizer as AddUserDataVectorizerNew
from embeddings.config import EmbeddingConfig
from embeddings.providers import EmbeddingProviderFactory

logger = CustomLogger().get_logger("embedding_migration")


class EmbeddingMigrationManager:
    def __init__(self):
        self.client = MongoClient(AppConfig.MONGODB_URI)
        self.db = self.client[AppConfig.DB_NAME]
        self.collection = self.db[AppConfig.COLLECTION_NAME]
        self.search_index_manager = SearchIndexManager()

        # Initialize new embedding manager with BAAI model
        self.new_embedding_config = EmbeddingConfig(
            provider="sentence_transformer",
            model_name="BAAI/bge-large-zh-v1.5",
            embedding_dimension=1024,
            device="cpu",  # Change to "cuda" if you have GPU
        )

        provider = EmbeddingProviderFactory.create_provider(
            provider_type=self.new_embedding_config.provider,
            model_name=self.new_embedding_config.model_name,
            device=self.new_embedding_config.device,
        )

        self.embedding_manager = EmbeddingManager(provider)
        self.resume_vectorizer = ResumeVectorizer(self.embedding_manager)
        self.add_user_data_vectorizer = AddUserDataVectorizerNew(self.embedding_manager)

    def backup_collection(self, backup_suffix: str = None) -> str:
        """Create a backup of the current collection"""
        if backup_suffix is None:
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

        backup_collection_name = f"{AppConfig.COLLECTION_NAME}_backup_{backup_suffix}"

        try:
            logger.info(f"Creating backup collection: {backup_collection_name}")

            # Use aggregation pipeline to copy data
            pipeline = [{"$out": backup_collection_name}]
            list(self.collection.aggregate(pipeline))

            # Verify backup
            original_count = self.collection.count_documents({})
            backup_count = self.db[backup_collection_name].count_documents({})

            if original_count == backup_count:
                logger.info(
                    f"Backup successful: {original_count} documents copied to {backup_collection_name}"
                )
                return backup_collection_name
            else:
                raise Exception(
                    f"Backup verification failed: {original_count} != {backup_count}"
                )

        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            raise

    def update_search_index(self) -> bool:
        """Update MongoDB search index to support 1024 dimensions"""
        try:
            logger.info("Updating search index for 1024 dimensions...")

            # Get new index definition
            new_index_def = {
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

            # First, try to delete existing index
            logger.info("Attempting to delete existing search index...")
            delete_success, delete_result = (
                self.search_index_manager.delete_search_index("vector_search_index")
            )

            if not delete_success:
                logger.warning(
                    f"Could not delete existing index (may not exist): {delete_result}"
                )
                # Continue anyway - the index might not exist yet
            else:
                logger.info("Existing search index deleted successfully")

            # Create the new index
            logger.info("Creating new search index...")
            create_success, create_result = (
                self.search_index_manager.create_search_index(new_index_def)
            )

            if create_success:
                logger.info("Search index created successfully")
                return True
            else:
                logger.error(f"Failed to create search index: {create_result}")
                return False

        except Exception as e:
            logger.error(f"Error updating search index: {str(e)}")
            return False

    def identify_document_type(self, doc: Dict) -> str:
        """Identify if document is regular resume or add_userdata format"""
        # Check for add_userdata specific fields
        if "academic_details" in doc and "may_also_known_skills" in doc:
            return "add_userdata"
        else:
            return "regular_resume"

    def regenerate_document_embeddings(self, doc: Dict) -> Dict:
        """Regenerate embeddings for a single document with new model"""
        try:
            doc_type = self.identify_document_type(doc)

            if doc_type == "add_userdata":
                # Use AddUserDataVectorizer
                updated_doc = self.add_user_data_vectorizer.generate_resume_embeddings(
                    doc
                )
            else:
                # Use ResumeVectorizer
                updated_doc = self.resume_vectorizer.generate_resume_embeddings(doc)

            logger.debug(
                f"Regenerated embeddings for document {doc.get('_id', 'unknown')}"
            )
            return updated_doc

        except Exception as e:
            logger.error(
                f"Error regenerating embeddings for document {doc.get('_id', 'unknown')}: {str(e)}"
            )
            return doc

    def migrate_all_embeddings(self, batch_size: int = 100) -> bool:
        """Migrate all document embeddings in batches"""
        try:
            total_docs = self.collection.count_documents({})
            logger.info(f"Starting migration of {total_docs} documents...")

            processed = 0
            failed = 0

            # Process documents in batches
            for i in range(0, total_docs, batch_size):
                logger.info(
                    f"Processing batch {i//batch_size + 1}: documents {i+1} to {min(i+batch_size, total_docs)}"
                )

                # Get batch of documents
                batch_docs = list(self.collection.find({}).skip(i).limit(batch_size))

                for doc in batch_docs:
                    try:
                        # Regenerate embeddings
                        updated_doc = self.regenerate_document_embeddings(doc)

                        # Update document in database
                        self.collection.replace_one({"_id": doc["_id"]}, updated_doc)

                        processed += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to process document {doc.get('_id', 'unknown')}: {str(e)}"
                        )
                        failed += 1

                # Log progress
                logger.info(
                    f"Batch completed. Processed: {processed}, Failed: {failed}"
                )

            logger.info(
                f"Migration completed. Total processed: {processed}, Failed: {failed}"
            )
            return failed == 0

        except Exception as e:
            logger.error(f"Error during migration: {str(e)}")
            return False

    def verify_migration(self, sample_size: int = 10) -> bool:
        """Verify that embeddings have correct dimensions"""
        try:
            logger.info(f"Verifying migration with {sample_size} sample documents...")

            sample_docs = list(self.collection.find({}).limit(sample_size))

            for doc in sample_docs:
                # Check vector fields
                vector_fields = [
                    "combined_resume_vector",
                    "skills_vector",
                    "experience_text_vector",
                    "academic_details_vector",
                ]

                for field in vector_fields:
                    if field in doc and doc[field]:
                        if len(doc[field]) != 1024:
                            logger.error(
                                f"Document {doc['_id']} has incorrect dimension for {field}: {len(doc[field])}"
                            )
                            return False

            logger.info(
                "Migration verification successful - all vectors have 1024 dimensions"
            )
            return True

        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            return False

    def update_env_config(self) -> bool:
        """Update environment configuration to use new model"""
        try:
            env_file_path = os.path.join(os.path.dirname(__file__), ".env")

            if not os.path.exists(env_file_path):
                logger.warning(f".env file not found at {env_file_path}")
                return False

            # Read current .env file
            with open(env_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Update relevant lines
            updated_lines = []
            for line in lines:
                if line.startswith("SENTENCE_TRANSFORMER_MODEL="):
                    updated_lines.append(
                        "SENTENCE_TRANSFORMER_MODEL=BAAI/bge-large-zh-v1.5\n"
                    )
                elif line.startswith("EMBEDDING_DIMENSIONS="):
                    updated_lines.append("EMBEDDING_DIMENSIONS=1024\n")
                else:
                    updated_lines.append(line)

            # Write updated .env file
            with open(env_file_path, "w", encoding="utf-8") as f:
                f.writelines(updated_lines)

            logger.info("Environment configuration updated successfully")
            return True

        except Exception as e:
            logger.error(f"Error updating environment configuration: {str(e)}")
            return False


def main():
    """Main migration function"""
    logger.info(
        "Starting embedding migration to BAAI/bge-large-zh-v1.5 (1024 dimensions)"
    )

    migration_manager = EmbeddingMigrationManager()

    try:
        # Step 1: Create backup
        logger.info("Step 1: Creating backup...")
        backup_name = migration_manager.backup_collection()
        logger.info(f"Backup created: {backup_name}")

        # Step 2: Update search index
        logger.info("Step 2: Updating search index...")
        if not migration_manager.update_search_index():
            logger.error("Failed to update search index. Migration aborted.")
            return False

        # Step 3: Migrate embeddings
        logger.info("Step 3: Migrating document embeddings...")
        if not migration_manager.migrate_all_embeddings():
            logger.error("Migration completed with errors. Check logs for details.")

        # Step 4: Verify migration
        logger.info("Step 4: Verifying migration...")
        if not migration_manager.verify_migration():
            logger.error("Migration verification failed!")
            return False

        # Step 5: Update environment configuration
        logger.info("Step 5: Updating environment configuration...")
        migration_manager.update_env_config()

        logger.info("✅ Migration completed successfully!")
        logger.info(f"Backup collection: {backup_name}")
        logger.info("Next steps:")
        logger.info("1. Test your application with the new embeddings")
        logger.info("2. If everything works, you can delete the backup collection")
        logger.info("3. Restart your application to use the new configuration")

        return True

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Migration completed successfully!")
    else:
        print("❌ Migration failed. Check logs for details.")
