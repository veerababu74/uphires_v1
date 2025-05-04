# resume_api/api/city_search.py
from fastapi import APIRouter, Body, Query, HTTPException
from typing import List, Dict, Any
from database.client import get_collection
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
    Search for resumes where the candidate's address contains the specified city.
    
    **Input:**
    - city: Name of the city to search for
    - limit: Maximum number of resumes to return
    
    **Returns:**
    List of matching resume profiles with contact details and other information
    
    **Example Response:**
    ```json
    [
        {
            "name": "John Doe",
            "contact_details": {
                "email": "john@example.com",
                "phone": "+91-1234567890",
                "address": "123 Main St, Mumbai 400001"
            },
            "education": [...],
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
                            "name": "John Doe",
                            "contact_details": {
                                "email": "john@example.com",
                                "phone": "+91-1234567890",
                                "address": "123 Main St, Mumbai 400001",
                            },
                            "education": [
                                {
                                    "degree": "B.Tech",
                                    "institution": "IIT Mumbai",
                                    "year": "2020",
                                }
                            ],
                            "experience": [
                                {
                                    "title": "Software Engineer",
                                    "company": "Tech Corp",
                                    "duration": "2020-Present",
                                }
                            ],
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
    city: str = Body(..., description="City name to search for in address"),
    limit: int = Body(10, description="Maximum number of results to return"),
):
    """
    Search for resumes where the contact address contains the specified city name.
    Returns all resumes that match the city name in the address field.
    """
    try:
        # Case-insensitive search for city name in address field
        city_pattern = re.compile(f"\\b{re.escape(city)}\\b", re.IGNORECASE)

        # Search resumes where address contains the city name
        query = {"contact_details.address": {"$regex": city_pattern}}

        # Get matching resumes
        results = list(collection.find(query).limit(limit))

        # Format results
        formatted_results = [format_resume(result) for result in results]

        # If no exact matches found, try to use the extract_city function to match on cities
        if not formatted_results:
            # Get all resumes
            all_resumes = list(
                collection.find({}).limit(100)
            )  # Limit to prevent too many processing

            # Filter resumes where extracted city matches the search city
            matching_resumes = []
            for resume in all_resumes:
                address = resume.get("contact_details", {}).get("address", "")
                extracted_city = extract_city(address)

                if extracted_city and city_pattern.search(extracted_city):
                    matching_resumes.append(resume)

                if len(matching_resumes) >= limit:
                    break

            # Format matches
            formatted_results = [format_resume(result) for result in matching_resumes]

        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"City search failed: {str(e)}")


# Improved extract_city function with better pattern matching
def extract_city(address):
    """
    Extract city name from an address string using multiple patterns and approaches.

    Args:
        address (str): Address string containing city name

    Returns:
        str or None: Extracted city name or None if not found
    """
    if not address or address == "N/A":
        return None

    # Pattern for "City, State" or "City, ST" format
    city_state_pattern = re.search(
        r"([A-Za-z\s]+),\s*([A-Za-z]{2}|[A-Za-z\s]+)", address
    )
    if city_state_pattern:
        return city_state_pattern.group(1).strip()

    # Pattern for city after numeric (e.g., "123 Main St, City")
    after_street_pattern = re.search(r"\d+[^,]+,\s*([A-Za-z\s]+)", address)
    if after_street_pattern:
        return after_street_pattern.group(1).strip()

    # Pattern for Indian addresses: typically City appears before PIN code
    indian_pattern = re.search(r"([A-Za-z\s]+)[\s-]*\d{6}", address)
    if indian_pattern:
        return indian_pattern.group(1).strip()

    # Fallback: Split by comma and take second part (if exists)
    parts = address.split(",")
    if len(parts) > 1:
        return parts[1].strip()

    # Last resort: check if address matches any city in the Indian cities list
    for city in cities:  # Using the indiancities.json loaded at startup
        if re.search(r"\b" + re.escape(city) + r"\b", address, re.IGNORECASE):
            return city

    return None
