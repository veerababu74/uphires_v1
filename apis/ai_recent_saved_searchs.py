from mangodatabase.client import (
    get_ai_recent_search_collection,
    get_ai_saved_searches_collection,
)

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid

router = APIRouter(
    prefix="/ai_search_operations",
    tags=["AI Search Save & Recent"],
)


class AISearchPayload(BaseModel):
    """Model for AI search payload to be saved"""

    user_id: str = Field(..., description="User ID who performed the search")
    query: str = Field(
        ...,
        description="AI search query",
        example="Find software engineers with Python experience",
    )


class SavedSearchResponse(BaseModel):
    """Response model for saved search operations"""

    success: bool
    message: str
    search_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class RecentSearchResponse(BaseModel):
    """Response model for recent searches"""

    success: bool
    searches: List[Dict[str, Any]]
    total_count: int


@router.post(
    "/save_search",
    response_model=SavedSearchResponse,
    summary="Save AI Search Query",
    description="Save an AI search query to the saved searches collection",
)
async def save_ai_search(payload: AISearchPayload):
    """Save AI search payload to AI saved searches collection"""
    try:
        collection = get_ai_saved_searches_collection()

        # Generate unique search ID
        search_id = str(uuid.uuid4())

        # Create search document
        search_document = {
            "search_id": search_id,
            "user_id": payload.user_id,
            "query": payload.query,
            "timestamp": datetime.now(timezone.utc),
            "search_type": "ai",
            "is_saved": True,
        }

        # Insert document into collection
        result = collection.insert_one(search_document)

        if result.inserted_id:
            return SavedSearchResponse(
                success=True,
                message="AI search saved successfully",
                search_id=search_id,
                timestamp=search_document["timestamp"],
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save AI search")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving AI search: {str(e)}")


@router.post(
    "/save_recent_search",
    response_model=SavedSearchResponse,
    summary="Save AI Search to Recent",
    description="Save an AI search query to the recent searches collection",
)
async def save_recent_ai_search(payload: AISearchPayload):
    """Save AI search payload to AI recent searches collection"""
    try:
        collection = get_ai_recent_search_collection()

        # Generate unique search ID
        search_id = str(uuid.uuid4())

        # Create search document
        search_document = {
            "search_id": search_id,
            "user_id": payload.user_id,
            "query": payload.query,
            "timestamp": datetime.now(timezone.utc),
            "search_type": "ai",
            "is_recent": True,
        }

        # Check if user has too many recent searches (limit to 100)
        user_recent_count = collection.count_documents({"user_id": payload.user_id})
        if user_recent_count >= 100:
            # Remove oldest search for this user
            oldest_search = collection.find_one(
                {"user_id": payload.user_id}, sort=[("timestamp", 1)]
            )
            if oldest_search:
                collection.delete_one({"_id": oldest_search["_id"]})

        # Insert new recent search
        result = collection.insert_one(search_document)

        if result.inserted_id:
            return SavedSearchResponse(
                success=True,
                message="Recent AI search saved successfully",
                search_id=search_id,
                timestamp=search_document["timestamp"],
            )
        else:
            raise HTTPException(
                status_code=500, detail="Failed to save recent AI search"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error saving recent AI search: {str(e)}"
        )


@router.get(
    "/saved_searches/{user_id}",
    response_model=RecentSearchResponse,
    summary="Get User's Saved AI Searches",
    description="Retrieve saved AI searches for a specific user with optional limit",
)
async def get_saved_ai_searches(user_id: str, limit: Optional[int] = None):
    """Get saved AI searches for a user with optional limit"""
    try:
        collection = get_ai_saved_searches_collection()

        # Build query with optional limit
        query = {"user_id": user_id}
        cursor = collection.find(query, sort=[("timestamp", -1)])

        if limit is not None and limit > 0:
            cursor = cursor.limit(limit)

        searches = list(cursor)

        # Convert ObjectId to string for JSON serialization
        for search in searches:
            search["_id"] = str(search["_id"])

        return RecentSearchResponse(
            success=True, searches=searches, total_count=len(searches)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving saved AI searches: {str(e)}"
        )


@router.get(
    "/recent_searches/{user_id}",
    response_model=RecentSearchResponse,
    summary="Get User's Recent AI Searches",
    description="Retrieve recent AI searches for a specific user with optional limit (default: 10, max: 50)",
)
async def get_recent_ai_searches(user_id: str, limit: Optional[int] = 10):
    """Get recent AI searches for a user with optional limit"""
    try:
        collection = get_ai_recent_search_collection()

        # Validate and set limit (default 10, max 50)
        if limit is None:
            limit = 10
        elif limit > 50:
            limit = 50
        elif limit < 1:
            limit = 10

        # Find recent searches for the user, sorted by timestamp (newest first)
        searches = list(
            collection.find({"user_id": user_id}, sort=[("timestamp", -1)], limit=limit)
        )

        # Convert ObjectId to string for JSON serialization
        for search in searches:
            search["_id"] = str(search["_id"])

        return RecentSearchResponse(
            success=True, searches=searches, total_count=len(searches)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving recent AI searches: {str(e)}"
        )


@router.delete(
    "/saved_searches/{search_id}",
    response_model=SavedSearchResponse,
    summary="Delete Saved AI Search",
    description="Delete a specific saved AI search by search_id",
)
async def delete_saved_ai_search(search_id: str):
    """Delete a saved AI search"""
    try:
        collection = get_ai_saved_searches_collection()

        result = collection.delete_one({"search_id": search_id})

        if result.deleted_count > 0:
            return SavedSearchResponse(
                success=True, message="Saved AI search deleted successfully"
            )
        else:
            raise HTTPException(status_code=404, detail="Saved AI search not found")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting saved AI search: {str(e)}"
        )


@router.delete(
    "/recent_searches/{search_id}",
    response_model=SavedSearchResponse,
    summary="Delete Recent AI Search",
    description="Delete a specific recent AI search by search_id",
)
async def delete_recent_ai_search(search_id: str):
    """Delete a recent AI search"""
    try:
        collection = get_ai_recent_search_collection()

        result = collection.delete_one({"search_id": search_id})

        if result.deleted_count > 0:
            return SavedSearchResponse(
                success=True, message="Recent AI search deleted successfully"
            )
        else:
            raise HTTPException(status_code=404, detail="Recent AI search not found")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting recent AI search: {str(e)}"
        )
