#!/usr/bin/env python3
"""
Flexible Embedding Migration Script

This script allows you to migrate to any supported embedding model.
It's safer and more flexible than the original script.

Supported migrations:
- To any of the configured models (384, 768, or 1024 dimensions)
- Safe backup and recovery
- Flexible search index handling
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
from embeddings.config import (
    EmbeddingConfig,
    get_config_by_name,
    list_available_configs,
)
from embeddings.providers import EmbeddingProviderFactory

logger = CustomLogger().get_logger("flexible_embedding_migration")


class FlexibleEmbeddingMigrationManager:
    def __init__(self, target_model_config_name: str):
        self.client = MongoClient(AppConfig.MONGODB_URI)
        self.db = self.client[AppConfig.DB_NAME]
        self.collection = self.db[AppConfig.COLLECTION_NAME]
        self.search_index_manager = SearchIndexManager()

        # Get target model configuration
        try:
            self.target_config = get_config_by_name(target_model_config_name)
            logger.info(f"Target model: {self.target_config.model_name}")
            logger.info(f"Target dimensions: {self.target_config.embedding_dimension}")
        except ValueError as e:
            logger.error(f"Invalid model configuration: {e}")
            raise

        # Initialize embedding manager with target model
        provider_kwargs = {
            "provider_type": self.target_config.provider,
            "model_name": self.target_config.model_name,
            "device": self.target_config.device,
        }

        # Add trust_remote_code if needed
        if (
            hasattr(self.target_config, "trust_remote_code")
            and self.target_config.trust_remote_code
        ):
            provider_kwargs["trust_remote_code"] = self.target_config.trust_remote_code

        provider = EmbeddingProviderFactory.create_provider(**provider_kwargs)

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

    def create_search_index_for_target(self) -> bool:
        """Create search index for target model dimensions"""
        try:
            target_dimensions = self.target_config.embedding_dimension
            logger.info(f"Creating search index for {target_dimensions} dimensions...")

            # Use a temporary name first
            index_name = f"vector_search_index_{target_dimensions}"

            new_index_def = {
                "name": index_name,
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "combined_resume_vector": {
                                "type": "knnVector",
                                "dimensions": target_dimensions,
                                "similarity": "cosine",
                            },
                            "skills_vector": {
                                "type": "knnVector",
                                "dimensions": target_dimensions,
                                "similarity": "cosine",
                            },
                            "experience_text_vector": {
                                "type": "knnVector",
                                "dimensions": target_dimensions,
                                "similarity": "cosine",
                            },
                            "academic_details_vector": {
                                "type": "knnVector",
                                "dimensions": target_dimensions,
                                "similarity": "cosine",
                            },
                        },
                    }
                },
            }

            # Create the new index
            success, result = self.search_index_manager.create_search_index(
                new_index_def
            )

            if success:
                logger.info(f"Search index '{index_name}' created successfully")
                return True
            else:
                logger.error(f"Failed to create search index: {result}")
                return False

        except Exception as e:
            logger.error(f"Error creating search index: {str(e)}")
            return False

    def identify_document_type(self, doc: Dict) -> str:
        """Identify if document is regular resume or add_userdata format"""
        if "academic_details" in doc and "may_also_known_skills" in doc:
            return "add_userdata"
        else:
            return "regular_resume"

    def regenerate_document_embeddings(self, doc: Dict) -> Dict:
        """Regenerate embeddings for a single document with target model"""
        try:
            doc_type = self.identify_document_type(doc)

            if doc_type == "add_userdata":
                updated_doc = self.add_user_data_vectorizer.generate_resume_embeddings(
                    doc
                )
            else:
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

    def migrate_all_embeddings(self, batch_size: int = 10) -> bool:
        """Migrate all document embeddings in batches"""
        try:
            total_docs = self.collection.count_documents({})
            logger.info(f"Starting migration of {total_docs} documents...")

            processed = 0
            failed = 0

            for i in range(0, total_docs, batch_size):
                logger.info(
                    f"Processing batch {i//batch_size + 1}: documents {i+1} to {min(i+batch_size, total_docs)}"
                )

                batch_docs = list(self.collection.find({}).skip(i).limit(batch_size))

                for doc in batch_docs:
                    try:
                        updated_doc = self.regenerate_document_embeddings(doc)
                        self.collection.replace_one({"_id": doc["_id"]}, updated_doc)
                        processed += 1

                        if processed % 5 == 0:
                            logger.info(
                                f"Processed {processed}/{total_docs} documents..."
                            )

                    except Exception as e:
                        logger.error(
                            f"Failed to process document {doc.get('_id', 'unknown')}: {str(e)}"
                        )
                        failed += 1

                logger.info(
                    f"Batch {i//batch_size + 1} completed. Total processed: {processed}, Failed: {failed}"
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
            target_dimensions = self.target_config.embedding_dimension
            logger.info(
                f"Verifying migration with {sample_size} sample documents for {target_dimensions} dimensions..."
            )

            sample_docs = list(self.collection.find({}).limit(sample_size))

            for doc in sample_docs:
                vector_fields = [
                    "combined_resume_vector",
                    "skills_vector",
                    "experience_text_vector",
                    "academic_details_vector",
                ]

                for field in vector_fields:
                    if field in doc and doc[field]:
                        if len(doc[field]) != target_dimensions:
                            logger.error(
                                f"Document {doc['_id']} has incorrect dimension for {field}: {len(doc[field])}"
                            )
                            return False

            logger.info(
                f"Migration verification successful - all vectors have {target_dimensions} dimensions"
            )
            return True

        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            return False

    def update_env_config(self) -> bool:
        """Update environment configuration to use target model"""
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
                        f"SENTENCE_TRANSFORMER_MODEL={self.target_config.model_name}\n"
                    )
                elif line.startswith("EMBEDDING_DIMENSIONS="):
                    updated_lines.append(
                        f"EMBEDDING_DIMENSIONS={self.target_config.embedding_dimension}\n"
                    )
                elif line.startswith("TRUST_REMOTE_CODE=") and hasattr(
                    self.target_config, "trust_remote_code"
                ):
                    updated_lines.append(
                        f"TRUST_REMOTE_CODE={str(self.target_config.trust_remote_code).lower()}\n"
                    )
                else:
                    updated_lines.append(line)

            # Add TRUST_REMOTE_CODE if it doesn't exist and is needed
            if (
                hasattr(self.target_config, "trust_remote_code")
                and self.target_config.trust_remote_code
                and not any(
                    line.startswith("TRUST_REMOTE_CODE=") for line in updated_lines
                )
            ):
                # Find AI UTILITIES section and add it there
                for i, line in enumerate(updated_lines):
                    if "AI UTILITIES CONFIGURATION" in line:
                        # Insert after the section header
                        for j in range(i + 1, len(updated_lines)):
                            if updated_lines[j].startswith("EMBEDDING_PROVIDER="):
                                updated_lines.insert(
                                    j + 3,
                                    f"TRUST_REMOTE_CODE={str(self.target_config.trust_remote_code).lower()}\n",
                                )
                                break
                        break

            # Write updated .env file
            with open(env_file_path, "w", encoding="utf-8") as f:
                f.writelines(updated_lines)

            logger.info("Environment configuration updated successfully")
            return True

        except Exception as e:
            logger.error(f"Error updating environment configuration: {str(e)}")
            return False


def show_available_models():
    """Show available model configurations"""
    print("📋 Available Model Configurations:")
    print("=" * 50)

    configs = list_available_configs()
    for name, config in configs.items():
        dimensions = config["embedding_dimension"]
        model_name = config["model_name"]
        special = ""
        if config.get("trust_remote_code"):
            special = " (requires trust_remote_code=True)"

        print(f"  {name}:")
        print(f"    Model: {model_name}")
        print(f"    Dimensions: {dimensions}{special}")
        print()


def main():
    """Main migration function"""
    print("🚀 Flexible Embedding Migration Tool")
    print("=" * 50)

    # Show available models
    show_available_models()

    # Get user choice
    print("Which model do you want to migrate to?")
    print(
        "Enter the configuration name (e.g., 'e5-small-v2', 'nomic-embed-text-v1', 'baai-bge-large-zh'):"
    )

    # For automation, you can uncomment one of these:
    # target_model = "e5-small-v2"  # No migration needed if currently using all-MiniLM-L6-v2
    # target_model = "nomic-embed-text-v1"  # 768 dimensions
    # target_model = "baai-bge-large-zh"  # 1024 dimensions

    # Interactive mode
    target_model = input("Model choice: ").strip()

    if not target_model:
        print("❌ No model specified. Exiting.")
        return False

    try:
        migration_manager = FlexibleEmbeddingMigrationManager(target_model)
    except Exception as e:
        print(f"❌ Error initializing migration: {e}")
        return False

    logger.info(
        f"Starting migration to {migration_manager.target_config.model_name} ({migration_manager.target_config.embedding_dimension} dimensions)"
    )

    try:
        # Step 1: Create backup
        logger.info("Step 1: Creating backup...")
        backup_name = migration_manager.backup_collection()
        logger.info(f"Backup created: {backup_name}")

        # Step 2: Create search index for target dimensions
        logger.info("Step 2: Creating search index for target model...")
        if not migration_manager.create_search_index_for_target():
            logger.error("Failed to create search index. Migration aborted.")
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
        logger.info(f"Target model: {migration_manager.target_config.model_name}")
        logger.info(
            f"Target dimensions: {migration_manager.target_config.embedding_dimension}"
        )
        logger.info("")
        logger.info("🔧 NEXT STEPS:")
        logger.info(
            "1. In MongoDB Atlas, rename the new index to 'vector_search_index'"
        )
        logger.info("2. Delete the old search index if dimensions changed")
        logger.info("3. Test your application with the new embeddings")
        logger.info("4. If everything works, delete the backup collection")
        logger.info("5. Restart your application")

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
