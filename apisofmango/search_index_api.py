from fastapi import FastAPI, HTTPException, status, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import sys
import os

from contextlib import asynccontextmanager

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from properties.mango import MONGODB_URI, DB_NAME, COLLECTION_NAME
from mangodatabase.search_indexes import SearchIndexManager, initialize_database
from core.custom_logger import CustomLogger
from schemas.search_index_api_schemas import (
    CreateSearchIndexRequest,
    UpdateSearchIndexRequest,
    AddFieldRequest,
    UpdateFieldRequest,
)

logger = CustomLogger().get_logger("search_index_api")


# Global search index manager - will be set during application startup
search_index_manager = None


def set_search_index_manager(manager):
    """Set the global search index manager instance"""
    global search_index_manager
    search_index_manager = manager


def check_search_index_manager():
    """Check if search index manager is available"""
    if search_index_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Search index manager not initialized. Please wait for application startup to complete.",
        )


router = APIRouter(prefix="/search-indexes", tags=["Search Indexes crud"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        check_search_index_manager()
        success, indexes = search_index_manager.list_search_indexes()
        if success:
            return {
                "status": "healthy",
                "database": "connected",
                "indexes_count": len(indexes),
            }
        else:
            return {"status": "unhealthy", "error": indexes}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/search-indexes")
async def list_search_indexes():
    """List all search indexes"""
    try:
        check_search_index_manager()
        success, result = search_index_manager.list_search_indexes()
        if success:
            return {"success": True, "indexes": result}
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to list indexes: {result}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing search indexes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-indexes/{index_name}")
async def check_search_index(index_name: str):
    """Check if a specific search index exists"""
    try:
        check_search_index_manager()
        exists = search_index_manager.check_search_index_exists(index_name)
        success, indexes = search_index_manager.list_search_indexes()

        if success:
            index_details = None
            for index in indexes:
                if index.get("name") == index_name:
                    index_details = index
                    break

            return {
                "index_name": index_name,
                "exists": exists,
                "details": index_details,
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to check index: {indexes}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking search index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-indexes")
async def create_search_index(request: CreateSearchIndexRequest):
    """Create a new search index"""
    try:
        check_search_index_manager()
        # Check if index already exists
        if search_index_manager.check_search_index_exists(request.name):
            raise HTTPException(
                status_code=400, detail=f"Search index '{request.name}' already exists"
            )

        # Format the index definition
        index_definition = {"name": request.name, "definition": request.definition}

        success, result = search_index_manager.create_search_index(index_definition)
        if success:
            return {
                "success": True,
                "message": f"Search index '{request.name}' created successfully",
                "result": result,
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to create index: {result}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating search index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/search-indexes/{index_name}")
async def delete_search_index(index_name: str):
    """Delete a search index"""
    try:
        check_search_index_manager()
        # Check if index exists
        if not search_index_manager.check_search_index_exists(index_name):
            raise HTTPException(
                status_code=404, detail=f"Search index '{index_name}' not found"
            )

        success, result = search_index_manager.delete_search_index(index_name)
        if success:
            return {
                "success": True,
                "message": f"Search index '{index_name}' deleted successfully",
                "result": result,
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete index: {result}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting search index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/search-indexes/{index_name}")
async def update_search_index(index_name: str, request: UpdateSearchIndexRequest):
    """Update a search index (deletes and recreates)"""
    try:
        check_search_index_manager()
        # Check if index exists
        if not search_index_manager.check_search_index_exists(index_name):
            raise HTTPException(
                status_code=404, detail=f"Search index '{index_name}' not found"
            )

        # Format the new index definition
        new_index_definition = {
            "name": request.name,
            "definition": request.new_definition,
        }

        success, result = search_index_manager.update_search_index(
            index_name, new_index_definition
        )
        if success:
            return {
                "success": True,
                "message": f"Search index '{index_name}' updated successfully",
                "result": result,
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to update index: {result}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating search index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-indexes/{index_name}/fields")
async def add_field_to_index(index_name: str, request: AddFieldRequest):
    """Add a new field to an existing search index"""
    try:
        check_search_index_manager()
        # Get current index definition
        success, indexes = search_index_manager.list_search_indexes()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to retrieve indexes")

        current_index = None
        for index in indexes:
            if index.get("name") == index_name:
                current_index = index
                break

        if not current_index:
            raise HTTPException(
                status_code=404, detail=f"Search index '{index_name}' not found"
            )

        # Add new field to the definition
        new_definition = current_index.copy()
        fields = new_definition["latestDefinition"]["mappings"]["fields"]
        fields[request.field_name] = {
            "type": request.field_definition.type,
            "dimensions": request.field_definition.dimensions,
            "similarity": request.field_definition.similarity,
        }

        # Update the index
        update_request = {
            "name": index_name,
            "definition": new_definition["latestDefinition"],
        }

        success, result = search_index_manager.update_search_index(
            index_name, update_request
        )
        if success:
            return {
                "success": True,
                "message": f"Field '{request.field_name}' added to index '{index_name}'",
                "result": result,
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to add field: {result}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding field to search index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/search-indexes/{index_name}/fields/{field_name}")
async def remove_field_from_index(index_name: str, field_name: str):
    """Remove a field from an existing search index"""
    try:
        check_search_index_manager()
        # Get current index definition
        success, indexes = search_index_manager.list_search_indexes()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to retrieve indexes")

        current_index = None
        for index in indexes:
            if index.get("name") == index_name:
                current_index = index
                break

        if not current_index:
            raise HTTPException(
                status_code=404, detail=f"Search index '{index_name}' not found"
            )

        # Remove field from the definition
        new_definition = current_index.copy()
        fields = new_definition["latestDefinition"]["mappings"]["fields"]

        if field_name not in fields:
            raise HTTPException(
                status_code=404,
                detail=f"Field '{field_name}' not found in index '{index_name}'",
            )

        del fields[field_name]

        # Update the index
        update_request = {
            "name": index_name,
            "definition": new_definition["latestDefinition"],
        }

        success, result = search_index_manager.update_search_index(
            index_name, update_request
        )
        if success:
            return {
                "success": True,
                "message": f"Field '{field_name}' removed from index '{index_name}'",
                "result": result,
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to remove field: {result}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing field from search index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/search-indexes/{index_name}/fields/{field_name}")
async def update_field_in_index(
    index_name: str, field_name: str, request: UpdateFieldRequest
):
    """Update a field in an existing search index"""
    try:
        # Get current index definition
        success, indexes = search_index_manager.list_search_indexes()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to retrieve indexes")

        current_index = None
        for index in indexes:
            if index.get("name") == index_name:
                current_index = index
                break

        if not current_index:
            raise HTTPException(
                status_code=404, detail=f"Search index '{index_name}' not found"
            )

        # Update field in the definition
        new_definition = current_index.copy()
        fields = new_definition["latestDefinition"]["mappings"]["fields"]

        if field_name not in fields:
            raise HTTPException(
                status_code=404,
                detail=f"Field '{field_name}' not found in index '{index_name}'",
            )

        fields[field_name] = {
            "type": request.new_field_definition.type,
            "dimensions": request.new_field_definition.dimensions,
            "similarity": request.new_field_definition.similarity,
        }

        # Update the index
        update_request = {
            "name": index_name,
            "definition": new_definition["latestDefinition"],
        }

        success, result = search_index_manager.update_search_index(
            index_name, update_request
        )
        if success:
            return {
                "success": True,
                "message": f"Field '{field_name}' updated in index '{index_name}'",
                "result": result,
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to update field: {result}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating field in search index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
