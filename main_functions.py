# main_functions.py
"""
Main application functions for database initialization and index creation.
This module contains the core functions used during application startup.
"""

from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("main_functions")


async def create_standard_indexes(collection, skills_titles_collection):
    """Create standard MongoDB indexes (non-Atlas Search)"""
    try:
        # Get existing indexes to avoid conflicts
        existing_indexes = list(collection.list_indexes())
        existing_index_names = [idx["name"] for idx in existing_indexes]

        logger.info("Creating standard MongoDB indexes...")

        # OPTIMIZED INDEX STRATEGY
        # MongoDB recommends keeping index count under 30 for optimal performance
        # We've reduced from 40+ indexes to essential ones only based on:
        # 1. Frequent query patterns in resume search
        # 2. Critical business requirements (user lookup, skill search, salary filtering)
        # 3. Compound indexes for common query combinations

        # Create only essential indexes based on frequent queries
        indexes_to_create = [
            "user_id",  # Essential for user identification
            "notice_period",  # Used in availability filtering
            "total_experience",  # Frequently used in searches
            "skills",  # Critical for skill-based searches
            "may_also_known_skills",  # Critical for skill-based searches
            "current_salary",  # Used in salary filtering
            "expected_salary",  # Used in salary filtering
            "contact_details.email",  # Essential for contact identification
            "contact_details.pan_card",  # Essential for unique identification
            "contact_details.current_city",  # Frequently used in location searches
            "contact_details.looking_for_jobs_in",  # Used in location preferences
            "experience.title",  # Frequently used in job title searches
            "contact_details.addhar_number",  # Essential for unique identification
            "contact_details.phone_number",  # Essential for contact identification
        ]

        for index_field in indexes_to_create:
            index_name = f"{index_field}_1"
            if index_name not in existing_index_names:
                try:
                    collection.create_index(index_field)
                    logger.info(f"Created index: {index_field}")
                except Exception as e:
                    logger.warning(f"Failed to create index {index_field}: {str(e)}")

        # Create only the most important compound indexes for performance
        compound_indexes = [
            # Essential compound indexes for frequent queries
            (
                [("contact_details.pan_card", 1), ("contact_details.email", 1)],
                "contact_details.pan_card_1_contact_details.email_1",
            ),
            # Additional essential identification compound indexes
            (
                [("contact_details.addhar_number", 1), ("contact_details.email", 1)],
                "contact_details.addhar_number_1_contact_details.email_1",
            ),
            (
                [("contact_details.phone_number", 1), ("contact_details.email", 1)],
                "contact_details.phone_number_1_contact_details.email_1",
            ),
            # Skills and experience - most frequently used combination
            ([("skills", 1), ("total_experience", 1)], "skills_1_total_experience_1"),
            (
                [("may_also_known_skills", 1), ("total_experience", 1)],
                "may_also_known_skills_1_total_experience_1",
            ),
            # Location and salary - frequently used for filtering
            (
                [("contact_details.current_city", 1), ("expected_salary", 1)],
                "contact_details.current_city_1_expected_salary_1",
            ),
            # Salary range queries - essential for salary filtering
            (
                [("current_salary", 1), ("expected_salary", 1)],
                "current_salary_1_expected_salary_1",
            ),
            # Job title and experience - common search combination
            (
                [("experience.title", 1), ("total_experience", 1)],
                "experience.title_1_total_experience_1",
            ),
        ]

        for index_spec, expected_name in compound_indexes:
            if expected_name not in existing_index_names:
                try:
                    collection.create_index(index_spec)
                    logger.info(f"Created compound index: {expected_name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to create compound index {expected_name}: {str(e)}"
                    )

        # Create optimized text index for full-text search - only essential fields
        text_index_exists = any(
            idx.get("key", {}).get("_fts") == "text" for idx in existing_indexes
        )
        if not text_index_exists:
            try:
                collection.create_index(
                    [
                        ("skills", "text"),
                        ("may_also_known_skills", "text"),
                        ("experience.title", "text"),
                        ("experience.company", "text"),
                    ]
                )
                logger.info("Created optimized text index for full-text search")
            except Exception as e:
                logger.warning(f"Failed to create text index: {str(e)}")

        logger.info("Standard MongoDB indexes creation completed")
        return True

    except Exception as e:
        logger.error(f"Error creating standard indexes: {str(e)}")
        return False


async def create_skills_collection_indexes(skills_titles_collection):
    """Create indexes for skills_titles_collection"""
    try:
        # Handle skills_titles_collection indexes
        skills_existing_indexes = list(skills_titles_collection.list_indexes())
        skills_existing_names = [idx["name"] for idx in skills_existing_indexes]

        logger.info("Creating skills collection indexes...")

        # Create indexes for skills_titles_collection
        skills_indexes = ["type", "value"]
        for index_field in skills_indexes:
            index_name = f"{index_field}_1"
            if index_name not in skills_existing_names:
                try:
                    skills_titles_collection.create_index(index_field)
                    logger.info(f"Created skills index: {index_field}")
                except Exception as e:
                    logger.warning(
                        f"Failed to create skills index {index_field}: {str(e)}"
                    )

        # Create compound index for skills_titles_collection
        compound_name = "type_1_value_1"
        if compound_name not in skills_existing_names:
            try:
                skills_titles_collection.create_index([("type", 1), ("value", 1)])
                logger.info("Created skills compound index: type_1_value_1")
            except Exception as e:
                logger.warning(f"Failed to create skills compound index: {str(e)}")

        # Create text index for skills_titles_collection
        skills_text_exists = any(
            idx.get("key", {}).get("_fts") == "text" for idx in skills_existing_indexes
        )
        if not skills_text_exists:
            try:
                skills_titles_collection.create_index([("value", "text")])
                logger.info("Created skills text index")
            except Exception as e:
                logger.warning(
                    f"Failed to create skills text index: {str(e)}"
                )  # Create unique compound index for skills_titles_collection (with custom name to avoid conflicts)
        unique_compound_name = "unique_type_value_idx"
        if unique_compound_name not in skills_existing_names:
            try:
                skills_titles_collection.create_index(
                    [("type", 1), ("value", 1)], unique=True, name=unique_compound_name
                )
                logger.info(
                    "Created unique compound index for skills_titles_collection"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create unique compound index (may already exist): {str(e)}"
                )

        logger.info("Skills collection indexes creation completed")
        return True

    except Exception as e:
        logger.error(f"Error creating skills collection indexes: {str(e)}")
        return False


async def create_manual_recent_search_indexes(manual_recent_search_collection):
    """Create indexes for manual_recent_search_collection"""
    try:
        # Get existing indexes to avoid conflicts
        existing_indexes = list(manual_recent_search_collection.list_indexes())
        existing_index_names = [idx["name"] for idx in existing_indexes]

        logger.info("Creating manual recent search collection indexes...")

        # Create essential single-field indexes for frequent queries
        indexes_to_create = [
            "user_id",  # Essential for user-based searches
            "timestamp",  # Essential for time-based queries and sorting
        ]

        for index_field in indexes_to_create:
            index_name = f"{index_field}_1"
            if index_name not in existing_index_names:
                try:
                    manual_recent_search_collection.create_index(index_field)
                    logger.info(f"Created manual recent search index: {index_field}")
                except Exception as e:
                    logger.warning(
                        f"Failed to create manual recent search index {index_field}: {str(e)}"
                    )

        # Create compound indexes for common query combinations
        compound_indexes = [
            # Most important: user_id with timestamp for user's recent searches
            (
                [
                    ("user_id", 1),
                    ("timestamp", -1),
                ],  # -1 for descending timestamp (newest first)
                "user_id_1_timestamp_-1",
            ),
            # Additional compound for potential queries
            (
                [("user_id", 1), ("timestamp", 1)],  # ascending timestamp if needed
                "user_id_1_timestamp_1",
            ),
        ]

        for index_spec, expected_name in compound_indexes:
            if expected_name not in existing_index_names:
                try:
                    manual_recent_search_collection.create_index(index_spec)
                    logger.info(
                        f"Created manual recent search compound index: {expected_name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create manual recent search compound index {expected_name}: {str(e)}"
                    )

        logger.info("Manual recent search collection indexes creation completed")
        return True

    except Exception as e:
        logger.error(
            f"Error creating manual recent search collection indexes: {str(e)}"
        )
        return False


async def create_manual_saved_search_indexes(manual_saved_search_collection):
    """Create indexes for manual_saved_search_collection"""
    try:
        # Get existing indexes to avoid conflicts
        existing_indexes = list(manual_saved_search_collection.list_indexes())
        existing_index_names = [idx["name"] for idx in existing_indexes]

        logger.info("Creating manual recent search collection indexes...")

        # Create essential single-field indexes for frequent queries
        indexes_to_create = [
            "user_id",  # Essential for user-based searches
            "timestamp",  # Essential for time-based queries and sorting
        ]

        for index_field in indexes_to_create:
            index_name = f"{index_field}_1"
            if index_name not in existing_index_names:
                try:
                    manual_saved_search_collection.create_index(index_field)
                    logger.info(f"Created manual saved search index: {index_field}")
                except Exception as e:
                    logger.warning(
                        f"Failed to create manual saved search index {index_field}: {str(e)}"
                    )

        # Create compound indexes for common query combinations
        compound_indexes = [
            # Most important: user_id with timestamp for user's recent searches
            (
                [
                    ("user_id", 1),
                    ("timestamp", -1),
                ],  # -1 for descending timestamp (newest first)
                "user_id_1_timestamp_-1",
            ),
            # Additional compound for potential queries
            (
                [("user_id", 1), ("timestamp", 1)],  # ascending timestamp if needed
                "user_id_1_timestamp_1",
            ),
        ]

        for index_spec, expected_name in compound_indexes:
            if expected_name not in existing_index_names:
                try:
                    manual_saved_search_collection.create_index(index_spec)
                    logger.info(
                        f"Created manual saved search compound index: {expected_name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create manual saved search compound index {expected_name}: {str(e)}"
                    )

        logger.info("Manual saved search collection indexes creation completed")
        return True

    except Exception as e:
        logger.error(f"Error creating manual saved search collection indexes: {str(e)}")
        return False


async def create_ai_recent_search_indexes(ai_recent_search_collection):
    """Create indexes for ai_recent_search_collection"""
    try:
        # Get existing indexes to avoid conflicts
        existing_indexes = list(ai_recent_search_collection.list_indexes())
        existing_index_names = [idx["name"] for idx in existing_indexes]

        logger.info("Creating AI recent search collection indexes...")

        # Create essential single-field indexes for frequent queries
        indexes_to_create = [
            "user_id",  # Essential for user-based searches
            "timestamp",  # Essential for time-based queries and sorting
        ]

        for index_field in indexes_to_create:
            index_name = f"{index_field}_1"
            if index_name not in existing_index_names:
                try:
                    ai_recent_search_collection.create_index(index_field)
                    logger.info(f"Created AI recent search index: {index_field}")
                except Exception as e:
                    logger.warning(
                        f"Failed to create AI recent search index {index_field}: {str(e)}"
                    )

        # Create compound indexes for common query combinations
        compound_indexes = [
            # Most important: user_id with timestamp for user's recent searches
            (
                [
                    ("user_id", 1),
                    ("timestamp", -1),
                ],  # -1 for descending timestamp (newest first)
                "user_id_1_timestamp_-1",
            ),
            # Additional compound for potential queries
            (
                [("user_id", 1), ("timestamp", 1)],  # ascending timestamp if needed
                "user_id_1_timestamp_1",
            ),
        ]

        for index_spec, expected_name in compound_indexes:
            if expected_name not in existing_index_names:
                try:
                    ai_recent_search_collection.create_index(index_spec)
                    logger.info(
                        f"Created AI recent search compound index: {expected_name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create AI recent search compound index {expected_name}: {str(e)}"
                    )

        logger.info("AI recent search collection indexes creation completed")
        return True

    except Exception as e:
        logger.error(f"Error creating AI recent search collection indexes: {str(e)}")
        return False


async def create_ai_saved_search_indexes(ai_saved_search_collection):
    """Create indexes for ai_saved_search_collection"""
    try:
        # Get existing indexes to avoid conflicts
        existing_indexes = list(ai_saved_search_collection.list_indexes())
        existing_index_names = [idx["name"] for idx in existing_indexes]

        logger.info("Creating AI saved search collection indexes...")

        # Create essential single-field indexes for frequent queries
        indexes_to_create = [
            "user_id",  # Essential for user-based searches
            "timestamp",  # Essential for time-based queries and sorting
        ]

        for index_field in indexes_to_create:
            index_name = f"{index_field}_1"
            if index_name not in existing_index_names:
                try:
                    ai_saved_search_collection.create_index(index_field)
                    logger.info(f"Created AI saved search index: {index_field}")
                except Exception as e:
                    logger.warning(
                        f"Failed to create AI saved search index {index_field}: {str(e)}"
                    )

        # Create compound indexes for common query combinations
        compound_indexes = [
            # Most important: user_id with timestamp for user's saved searches
            (
                [
                    ("user_id", 1),
                    ("timestamp", -1),
                ],  # -1 for descending timestamp (newest first)
                "user_id_1_timestamp_-1",
            ),
            # Additional compound for potential queries
            (
                [("user_id", 1), ("timestamp", 1)],  # ascending timestamp if needed
                "user_id_1_timestamp_1",
            ),
        ]

        for index_spec, expected_name in compound_indexes:
            if expected_name not in existing_index_names:
                try:
                    ai_saved_search_collection.create_index(index_spec)
                    logger.info(
                        f"Created AI saved search compound index: {expected_name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create AI saved search compound index {expected_name}: {str(e)}"
                    )

        logger.info("AI saved search collection indexes creation completed")
        return True

    except Exception as e:
        logger.error(f"Error creating AI saved search collection indexes: {str(e)}")
        return False


async def initialize_application_startup():
    """
    Initialize the application during startup.
    This function handles database initialization, connection testing, and index creation.

    Returns:
        tuple: (search_index_manager, success_status)
    """
    from mangodatabase.search_indexes import SearchIndexManager, initialize_database
    from mangodatabase.client import get_collection, get_skills_titles_collection
    from mangodatabase.client import (
        get_ai_recent_search_collection,
        get_ai_saved_searches_collection,
    )
    from mangodatabase.client import (
        get_manual_recent_search_collection,
        get_manual_saved_searches_collection,
    )

    logger.info("Starting up FastAPI application...")

    # Initialize database and search index manager
    try:
        # Initialize database (creates collection and default search index if needed)
        init_success = initialize_database()
        if not init_success:
            logger.error("Failed to initialize database during startup")
            raise Exception("Database initialization failed")

        # Initialize search index manager
        search_index_manager = SearchIndexManager()
        logger.info("Search index manager initialized successfully")

        # Initialize database collections
        collection = get_collection()
        skills_titles_collection = get_skills_titles_collection()
        ai_recent_search_collection = get_ai_recent_search_collection()
        ai_saved_searches_collection = get_ai_saved_searches_collection()
        manual_recent_search_collection = get_manual_recent_search_collection()
        manual_saved_searches_collection = get_manual_saved_searches_collection()

        # Test database connection first
        collection.database.client.admin.command("ping")
        logger.info("Connected to MongoDB successfully!")

        skills_titles_collection.database.client.admin.command("ping")
        logger.info("Connected to Skills & Titles collection successfully!")

        # Create standard MongoDB indexes
        await create_standard_indexes(collection, skills_titles_collection)

        # Create skills collection indexes
        await create_skills_collection_indexes(
            skills_titles_collection
        )  # Create manual recent search collection indexes
        await create_manual_recent_search_indexes(manual_recent_search_collection)

        # Create manual saved search collection indexes
        await create_manual_saved_search_indexes(manual_saved_searches_collection)

        # Create AI recent search collection indexes
        await create_ai_recent_search_indexes(ai_recent_search_collection)

        # Create AI saved search collection indexes
        await create_ai_saved_search_indexes(ai_saved_searches_collection)

        # Create AI recent search collection indexes
        await create_ai_recent_search_indexes(ai_recent_search_collection)

        # Create AI saved search collection indexes
        await create_ai_saved_search_indexes(ai_saved_searches_collection)

        logger.info("Application startup completed successfully!")

        return search_index_manager, True

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        logger.warning(
            "Some indexes may not have been created, but the application will continue to run"
        )
        raise e


def handle_application_shutdown():
    """Handle application shutdown procedures"""
    logger.info("Shutting down FastAPI application...")
    # Add any cleanup procedures here if needed
