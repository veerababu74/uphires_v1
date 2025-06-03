# resume_api/api/city_search.py
from fastapi import APIRouter, Body, Query, HTTPException
from typing import List, Dict, Any
from mangodatabase.client import get_collection
from core.helpers import format_resume
import json
import re

router = APIRouter(
    prefix="/search",
    tags=["City Search"],
)

# Initialize database collection
collection = get_collection()
with open("data/indiancities.json", "r") as f:
    cities = [city.strip() for city in json.load(f)]


@router.get(
    "/indian_cities/",
    response_model=List[str],
    summary="Autocomplete Indian City Names",
    description="""
    Get autocomplete suggestions for Indian city names based on input prefix.
    
    **Parameters:**
    - q: Search prefix for city name (e.g., "mum" for "Mumbai")
    - limit: Maximum number of suggestions to return
    
    **Returns:**
    List of matching city names sorted alphabetically
    """,
    responses={
        200: {
            "description": "Successful city suggestions",
            "content": {
                "application/json": {"example": ["Mumbai", "Mumbra", "Mundra"]}
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search query cannot be empty"}
                }
            },
        },
    },
)
async def autocomplete_cities(
    q: str = Query(
        ..., description="City name prefix to search for", min_length=1, example="mum"
    ),
    limit: int = Query(
        default=10,
        description="Maximum number of suggestions to return",
        ge=1,
        le=50,
        example=10,
    ),
):
    """
    Get autocomplete suggestions for Indian city names.

    Args:
        q (str): Search prefix for city name
        limit (int): Maximum number of results to return

    Returns:
        List[str]: Matching city names
    """
    if not q:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    q_lower = q.lower()
    filtered_cities = [city for city in cities if city.lower().startswith(q_lower)]
    return sorted(filtered_cities[:limit])


@router.post(
    "/city/",
    response_model=List[Dict[str, Any]],
    summary="Search Resumes by City",
    description="""
    Search for resumes where the candidate's current city or preferred job locations contain the specified city.
    
    **Input:**
    - city: Name of the city to search for
    - limit: Maximum number of resumes to return
    
    **Search Logic:**
    - Searches in both 'current_city' and 'looking_for_jobs_in' fields
    - Returns resumes where city matches either current location or preferred job locations
    
    **Example Response:**
    ```json
    [
        {
            "user_id": "user123",
            "username": "johndoe",
            "contact_details": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "+91-1234567890",
                "current_city": "Mumbai",
                "looking_for_jobs_in": ["Mumbai", "Pune"],
                "pan_card": "ABCDE1234F"
            },
            "academic_details": [...],
            "experience": [...],
            "skills": [...]
        }
    ]
    ```
    """,
    responses={
        200: {
            "description": "Successful resume search results",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "user_id": "user123",
                            "username": "johndoe",
                            "contact_details": {
                                "name": "John Doe",
                                "email": "john@example.com",
                                "phone": "+91-1234567890",
                                "current_city": "Mumbai",
                                "looking_for_jobs_in": ["Mumbai", "Pune"],
                                "pan_card": "ABCDE1234F",
                            },
                            "academic_details": [
                                {
                                    "education": "B.Tech in Computer Science",
                                    "college": "IIT Mumbai",
                                    "pass_year": 2020,
                                }
                            ],
                            "experience": [
                                {
                                    "company": "Tech Corp",
                                    "title": "Software Engineer",
                                    "from_date": "2020-01",
                                    "to": "2023-12",
                                }
                            ],
                            "skills": ["Python", "React", "AWS"],
                            "may_also_known_skills": ["Docker", "Kubernetes"],
                        }
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {"example": {"detail": "City name cannot be empty"}}
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "City search failed: Database error"}
                }
            },
        },
    },
)
async def search_by_city(
    city: str = Body(
        ...,
        description="City name to search for in current_city or looking_for_jobs_in",
    ),
    limit: int = Body(10, description="Maximum number of results to return"),
):
    """
    Search for resumes where the current_city or looking_for_jobs_in contains the specified city name.
    Returns all resumes that match the city name in either current location or preferred job locations.
    """
    try:
        # Case-insensitive search for city name in current_city and looking_for_jobs_in fields
        city_pattern = re.compile(f"\\b{re.escape(city)}\\b", re.IGNORECASE)

        # Search resumes where current_city or looking_for_jobs_in contains the city name
        query = {
            "$or": [
                {"contact_details.current_city": {"$regex": city_pattern}},
                {"contact_details.looking_for_jobs_in": {"$regex": city_pattern}},
            ]
        }

        # Get matching resumes
        results = list(collection.find(query).limit(limit))

        # Format results
        formatted_results = [format_resume(result) for result in results]

        # If no exact matches found, try broader search
        if not formatted_results:
            # Search with case-insensitive contains (less strict)
            broader_query = {
                "$or": [
                    {
                        "contact_details.current_city": {
                            "$regex": re.compile(city, re.IGNORECASE)
                        }
                    },
                    {
                        "contact_details.looking_for_jobs_in": {
                            "$regex": re.compile(city, re.IGNORECASE)
                        }
                    },
                ]
            }

            broader_results = list(collection.find(broader_query).limit(limit))
            formatted_results = [format_resume(result) for result in broader_results]

        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"City search failed: {str(e)}")
