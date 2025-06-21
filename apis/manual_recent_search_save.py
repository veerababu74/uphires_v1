from mangodatabase.client import (
    get_manual_recent_search_collection,
    get_manual_saved_searches_collection,
)

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid

router = APIRouter(
    prefix="/manual_saved_recnet_search",
    tags=["Manual Search Save & Recent"],
)


class ManualSearchPayload(BaseModel):
    """Model for manual search payload to be saved"""

    userid: str = Field(..., description="User ID who performed the search")
    experience_titles: Optional[List[str]] = Field(
        default=None,
        description="List of job titles searched",
        example=["Software Engineer", "Python Developer"],
    )
    skills: Optional[List[str]] = Field(
        default=None,
        description="Technical skills searched",
        example=["Python", "React", "AWS"],
    )
    min_education: Optional[List[str]] = Field(
        default=None,
        description="Minimum education qualifications",
        example=["BTech", "BSc"],
    )
    min_experience: Optional[str] = Field(
        default=None,
        description="Minimum experience requirement",
        example="2 years 6 months",
    )
    max_experience: Optional[str] = Field(
        default=None, description="Maximum experience requirement", example="5 years"
    )
    locations: Optional[List[str]] = Field(
        default=None,
        description="Searched locations",
        example=["Mumbai", "Pune", "Bangalore"],
    )
    min_salary: Optional[float] = Field(
        default=None, description="Minimum salary filter", example=500000
    )
    max_salary: Optional[float] = Field(
        default=None, description="Maximum salary filter", example=1500000
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
    summary="Save Manual Search Query",
    description="Save a manual search query to the saved searches collection",
)
async def save_manual_search(payload: ManualSearchPayload):
    """Save manual search payload to manual saved searches collection"""
    try:
        collection = get_manual_saved_searches_collection()

        # Generate unique search ID
        search_id = str(uuid.uuid4())

        # Create search document
        search_document = {
            "search_id": search_id,
            "user_id": payload.userid,
            "search_criteria": {
                "experience_titles": payload.experience_titles,
                "skills": payload.skills,
                "min_education": payload.min_education,
                "min_experience": payload.min_experience,
                "max_experience": payload.max_experience,
                "locations": payload.locations,
                "min_salary": payload.min_salary,
                "max_salary": payload.max_salary,
            },
            "timestamp": datetime.now(timezone.utc),
            "search_type": "manual",
            "is_saved": True,
        }

        # Insert document into collection
        result = collection.insert_one(search_document)

        if result.inserted_id:
            return SavedSearchResponse(
                success=True,
                message="Search saved successfully",
                search_id=search_id,
                timestamp=search_document["timestamp"],
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save search")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving search: {str(e)}")


@router.post(
    "/save_recent_search",
    response_model=SavedSearchResponse,
    summary="Save Manual Search to Recent",
    description="Save a manual search query to the recent searches collection",
)
async def save_recent_manual_search(payload: ManualSearchPayload):
    """Save manual search payload to manual recent searches collection"""
    try:
        collection = get_manual_recent_search_collection()

        # Generate unique search ID
        search_id = str(uuid.uuid4())

        # Create search document
        search_document = {
            "search_id": search_id,
            "user_id": payload.userid,
            "search_criteria": {
                "experience_titles": payload.experience_titles,
                "skills": payload.skills,
                "min_education": payload.min_education,
                "min_experience": payload.min_experience,
                "max_experience": payload.max_experience,
                "locations": payload.locations,
                "min_salary": payload.min_salary,
                "max_salary": payload.max_salary,
            },
            "timestamp": datetime.now(timezone.utc),
            "search_type": "manual",
            "is_recent": True,
        }

        # Check if user has too many recent searches (limit to 100)
        user_recent_count = collection.count_documents({"user_id": payload.userid})
        if user_recent_count >= 100:
            # Remove oldest search for this user
            oldest_search = collection.find_one(
                {"user_id": payload.userid}, sort=[("timestamp", 1)]
            )
            if oldest_search:
                collection.delete_one({"_id": oldest_search["_id"]})

        # Insert new recent search
        result = collection.insert_one(search_document)

        if result.inserted_id:
            return SavedSearchResponse(
                success=True,
                message="Recent search saved successfully",
                search_id=search_id,
                timestamp=search_document["timestamp"],
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save recent search")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error saving recent search: {str(e)}"
        )


@router.get(
    "/saved_searches/{user_id}",
    response_model=RecentSearchResponse,
    summary="Get User's Saved Searches",
    description="Retrieve saved searches for a specific user with optional limit",
)
async def get_saved_searches(user_id: str, limit: Optional[int] = None):
    """Get saved searches for a user with optional limit"""
    try:
        collection = get_manual_saved_searches_collection()

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
            status_code=500, detail=f"Error retrieving saved searches: {str(e)}"
        )


@router.get(
    "/recent_searches/{user_id}",
    response_model=RecentSearchResponse,
    summary="Get User's Recent Searches",
    description="Retrieve recent searches for a specific user with optional limit (default: 10, max: 50)",
)
async def get_recent_searches(user_id: str, limit: Optional[int] = 10):
    """Get recent searches for a user with optional limit"""
    try:
        collection = get_manual_recent_search_collection()

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
            status_code=500, detail=f"Error retrieving recent searches: {str(e)}"
        )


@router.delete(
    "/saved_searches/{search_id}",
    response_model=SavedSearchResponse,
    summary="Delete Saved Search",
    description="Delete a specific saved search by search_id",
)
async def delete_saved_search(search_id: str):
    """Delete a saved search"""
    try:
        collection = get_manual_saved_searches_collection()

        result = collection.delete_one({"search_id": search_id})

        if result.deleted_count > 0:
            return SavedSearchResponse(
                success=True, message="Saved search deleted successfully"
            )
        else:
            raise HTTPException(status_code=404, detail="Saved search not found")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting saved search: {str(e)}"
        )


@router.delete(
    "/recent_searches/{search_id}",
    response_model=SavedSearchResponse,
    summary="Delete Recent Search",
    description="Delete a specific recent search by search_id",
)
async def delete_recent_search(search_id: str):
    """Delete a recent search"""
    try:
        collection = get_manual_recent_search_collection()

        result = collection.delete_one({"search_id": search_id})

        if result.deleted_count > 0:
            return SavedSearchResponse(
                success=True, message="Recent search deleted successfully"
            )
        else:
            raise HTTPException(status_code=404, detail="Recent search not found")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting recent search: {str(e)}"
        )
