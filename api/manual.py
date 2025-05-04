import re
from fastapi import APIRouter, Body, HTTPException, Query
from typing import List, Dict, Any, Optional
from database.client import get_collection
from core.helpers import format_resume
from core.vectorizer import Vectorizer

resumes_collection = get_collection()
vectorizer = Vectorizer()

router = APIRouter(prefix="/resume-search", tags=["Resume Search v1"])


@router.post("/semantic/", response_model=List[Dict[str, Any]])
async def semantic_resume_search(
    query: str = Body(..., description="Free text search query for the entire resume"),
    min_results: int = Body(10, description="Minimum number of results to return"),
):
    """
    Perform semantic vector search on the entire content of resumes based on total_resume_vector.
    Returns at least the specified minimum number of results.
    """
    try:
        # Generate embedding for search query
        query_embedding = vectorizer.generate_embedding(query)

        # Perform vector search using total_resume_vector
        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "total_resume_vector",
                        "k": min_results,
                    },
                }
            },
            {
                "$project": {
                    "name": 1,
                    "contact_details": 1,
                    "education": 1,
                    "experience": 1,
                    "projects": 1,
                    "total_experience": 1,
                    "skills": 1,
                    "certifications": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
        ]

        # Get results
        results = list(resumes_collection.aggregate(pipeline))

        # Format results
        formatted_results = []
        for resume in results:
            formatted_resume = format_resume(resume)
            if "total_resume_text" in formatted_resume:
                del formatted_resume["total_resume_text"]
            formatted_results.append(formatted_resume)

        return formatted_results

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Semantic resume search failed: {str(e)}"
        )


@router.post("/manual/", response_model=List[Dict[str, Any]])
async def manual_resume_search(
    experience_titles: List[str] = Body(
        ..., description="List of experience titles to search for (mandatory)"
    ),
    skills: Optional[List[str]] = Body(
        None, description="List of skills to search for"
    ),
    min_education: Optional[str] = Body(
        None,
        description="Minimum education level (e.g., '10th', 'Diploma', 'Degree', 'BTech')",
    ),
    min_experience: Optional[str] = Body(
        None, description="Minimum experience in format 'X years Y months'"
    ),
    locations: Optional[List[str]] = Body(
        None, description="List of city names to search in address"
    ),
    limit: int = Body(10, description="Maximum number of results to return"),
):
    """
    Manual search endpoint that combines multiple criteria:
    - Experience titles (mandatory)
    - Skills (optional)
    - Minimum education level (optional)
    - Minimum experience duration (optional)
    - Locations/cities (optional)

    Returns ranked results with the best matches first.
    """
    try:
        if not experience_titles:
            raise HTTPException(
                status_code=400, detail="At least one experience title is required"
            )

        # Parse min_experience string to months
        min_experience_months = 0
        if min_experience:
            min_experience_months = parse_experience_to_months(min_experience)

        # Step 1: Build the base query for experience titles
        title_patterns = [
            re.compile(f".*{re.escape(title)}.*", re.IGNORECASE)
            for title in experience_titles
        ]
        base_query = {
            "$or": [
                {"experience.title": {"$regex": pattern}} for pattern in title_patterns
            ]
        }

        # Step 2: Get results matching experience titles
        results = list(resumes_collection.find(base_query))

        # Step 3: Calculate scores and filter based on criteria
        filtered_results = []

        for resume in results:
            # Skip if we've reached the limit
            if len(filtered_results) >= limit:
                break

            match_score = 0
            should_include = True

            # Calculate experience title match score
            exp_title_matches = 0
            for exp in resume.get("experience", []):
                exp_title = exp.get("title", "").lower()
                for title in experience_titles:
                    if title.lower() in exp_title:
                        exp_title_matches += 1
                        break

            # Only include if at least one experience title matches
            if exp_title_matches == 0:
                continue

            match_score += exp_title_matches * 10  # Weight for title matches

            # Filter by skills if provided
            if skills and len(skills) > 0:
                resume_skills = [skill.lower() for skill in resume.get("skills", [])]
                skill_matches = sum(
                    1 for skill in skills if skill.lower() in resume_skills
                )

                # Add skill match score
                if skill_matches > 0:
                    match_score += (skill_matches / len(skills)) * 10
                else:
                    # Skip this resume if no skills match
                    continue

            # Filter by minimum education if provided
            if min_education:
                # Define education hierarchy
                education_levels = {
                    "10th": 1,
                    "ssc": 1,
                    "12th": 2,
                    "hsc": 2,
                    "inter": 2,
                    "intermediate": 2,
                    "diploma": 3,
                    "iti": 3,
                    "associate": 3,
                    "bachelor": 4,
                    "degree": 4,
                    "graduate": 4,
                    "btech": 4,
                    "be": 4,
                    "bsc": 4,
                    "ba": 4,
                    "bcom": 4,
                    "master": 5,
                    "post graduate": 5,
                    "mtech": 5,
                    "me": 5,
                    "msc": 5,
                    "ma": 5,
                    "mcom": 5,
                    "mba": 5,
                    "phd": 6,
                    "doctorate": 6,
                }

                min_level = education_levels.get(min_education.lower(), 0)
                if min_level > 0:  # Only apply filter if education level recognized
                    # Extract highest education level from resume
                    highest_level = 0
                    for edu in resume.get("education", []):
                        degree = edu.get("degree", "").lower()
                        # Check each word in the degree against the education levels
                        for level_name, level_value in education_levels.items():
                            if level_name in degree:
                                highest_level = max(highest_level, level_value)

                    if highest_level < min_level:
                        continue  # Skip if education requirement not met

                    match_score += highest_level  # Add score based on education level

            # Filter by minimum experience if provided
            if min_experience_months > 0:
                # Try to parse total_experience field
                total_exp = resume.get("total_experience", "")
                resume_exp_months = 0

                if total_exp and total_exp != "N/A":
                    resume_exp_months = parse_experience_to_months(total_exp)
                else:
                    # Calculate from individual experiences if total not available
                    for exp in resume.get("experience", []):
                        duration = exp.get("duration", "")
                        if duration and duration != "N/A":
                            exp_months = parse_experience_to_months(duration)
                            resume_exp_months += exp_months

                if resume_exp_months < min_experience_months:
                    continue  # Skip if experience requirement not met

                # Add score based on experience match
                if resume_exp_months <= min_experience_months * 1.5:
                    match_score += 10  # Closer to required experience
                else:
                    match_score += 5  # Much more than required experience

            # Filter by locations if provided
            if locations and len(locations) > 0:
                address = resume.get("contact_details", {}).get("address", "")
                location_match = False

                if address and address != "N/A":
                    # Simple check if any of the requested locations exist in the address
                    for location in locations:
                        if search_city_in_address(address, location):
                            location_match = True
                            match_score += 5  # Bonus for location match
                            break

                if not location_match and len(locations) > 0:
                    continue  # Skip if no location match and locations were specified

            # Add to filtered results with score
            formatted_resume = format_resume(resume)

            # Remove the total_resume_text field if present
            if "total_resume_text" in formatted_resume:
                del formatted_resume["total_resume_text"]

            formatted_resume["match_score"] = match_score
            filtered_results.append(formatted_resume)

        # Sort by match score
        sorted_results = sorted(
            filtered_results, key=lambda x: x.get("match_score", 0), reverse=True
        )

        # Return top results up to limit
        return sorted_results[:limit]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual search failed: {str(e)}")


@router.post("/hybrid/", response_model=List[Dict[str, Any]])
async def hybrid_resume_search(
    query: str = Body(..., description="Free text search query"),
    experience_titles: Optional[List[str]] = Body(
        None, description="List of experience titles to search for"
    ),
    skills: Optional[List[str]] = Body(
        None, description="List of skills to search for"
    ),
    min_education: Optional[str] = Body(
        None,
        description="Minimum education level (e.g., '10th', 'Diploma', 'Degree', 'BTech')",
    ),
    min_experience: Optional[str] = Body(
        None, description="Minimum experience in format 'X years Y months'"
    ),
    locations: Optional[List[str]] = Body(
        None, description="List of city names to search in address"
    ),
    limit: int = Body(10, description="Maximum number of results to return"),
):
    """
    Hybrid search endpoint that combines semantic vector search with structured filters:
    - Free text query (semantic search)
    - Experience titles (optional)
    - Skills (optional)
    - Minimum education level (optional)
    - Minimum experience duration (optional)
    - Locations/cities (optional)

    This provides the best of both worlds - accurate semantic matching with precise filtering.
    """
    try:
        # Generate embedding for search query
        query_embedding = vectorizer.generate_embedding(query)

        # Parse min_experience string to months if provided
        min_experience_months = 0
        if min_experience:
            min_experience_months = parse_experience_to_months(min_experience)

        # Build title regex patterns if provided
        title_filter = {}
        if experience_titles and len(experience_titles) > 0:
            title_patterns = [
                re.compile(f".*{re.escape(title)}.*", re.IGNORECASE)
                for title in experience_titles
            ]
            title_filter = {
                "$or": [
                    {"experience.title": {"$regex": pattern}}
                    for pattern in title_patterns
                ]
            }

        # First perform vector search to get initial candidates (get more than needed for filtering)
        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "total_resume_vector",
                        "k": limit * 5,  # Get more results than needed for filtering
                    },
                }
            },
            {
                "$project": {
                    "name": 1,
                    "contact_details": 1,
                    "education": 1,
                    "experience": 1,
                    "projects": 1,
                    "total_experience": 1,
                    "skills": 1,
                    "certifications": 1,
                    "semantic_score": {"$meta": "searchScore"},
                }
            },
        ]

        initial_results = list(resumes_collection.aggregate(pipeline))

        # Apply post-filters and calculate combined score
        filtered_results = []

        for resume in initial_results:
            # Skip if we've reached the limit
            if len(filtered_results) >= limit:
                break

            match_score = (
                resume.get("semantic_score", 0) * 10
            )  # Base score from semantic search
            should_include = True

            # Filter by experience titles if provided
            if experience_titles and len(experience_titles) > 0:
                # Check if any titles match
                title_match = False
                for exp in resume.get("experience", []):
                    exp_title = exp.get("title", "").lower()
                    for title in experience_titles:
                        if title.lower() in exp_title:
                            title_match = True
                            match_score += 10  # Bonus for title match
                            break
                    if title_match:
                        break

                if not title_match:
                    continue  # Skip if no title matches

            # Filter by skills if provided
            if skills and len(skills) > 0:
                resume_skills = [skill.lower() for skill in resume.get("skills", [])]
                skill_matches = sum(
                    1 for skill in skills if skill.lower() in resume_skills
                )

                if skill_matches > 0:
                    match_score += (
                        skill_matches / len(skills)
                    ) * 10  # Weight for skill matches
                else:
                    continue  # Skip if no skills match

            # Filter by minimum education if provided
            if min_education:
                # Define education hierarchy
                education_levels = {
                    "10th": 1,
                    "ssc": 1,
                    "12th": 2,
                    "hsc": 2,
                    "inter": 2,
                    "intermediate": 2,
                    "diploma": 3,
                    "iti": 3,
                    "associate": 3,
                    "bachelor": 4,
                    "degree": 4,
                    "graduate": 4,
                    "btech": 4,
                    "be": 4,
                    "bsc": 4,
                    "ba": 4,
                    "bcom": 4,
                    "master": 5,
                    "post graduate": 5,
                    "mtech": 5,
                    "me": 5,
                    "msc": 5,
                    "ma": 5,
                    "mcom": 5,
                    "mba": 5,
                    "phd": 6,
                    "doctorate": 6,
                }

                min_level = education_levels.get(min_education.lower(), 0)
                if min_level > 0:
                    # Extract highest education level from resume
                    highest_level = 0
                    for edu in resume.get("education", []):
                        degree = edu.get("degree", "").lower()
                        for level_name, level_value in education_levels.items():
                            if level_name in degree:
                                highest_level = max(highest_level, level_value)

                    if highest_level < min_level:
                        continue  # Skip if education requirement not met

                    match_score += highest_level  # Add score based on education level

            # Filter by minimum experience if provided
            if min_experience_months > 0:
                # Try to get total experience
                total_exp = resume.get("total_experience", "")
                resume_exp_months = 0

                if total_exp and total_exp != "N/A":
                    resume_exp_months = parse_experience_to_months(total_exp)
                else:
                    # Calculate from individual experiences
                    for exp in resume.get("experience", []):
                        duration = exp.get("duration", "")
                        if duration and duration != "N/A":
                            exp_months = parse_experience_to_months(duration)
                            resume_exp_months += exp_months

                if resume_exp_months < min_experience_months:
                    continue  # Skip if experience requirement not met

                # Add score based on experience match
                if resume_exp_months <= min_experience_months * 1.5:
                    match_score += 10  # Closer to required experience
                else:
                    match_score += 5  # Much more than required experience

            # Filter by locations if provided
            if locations and len(locations) > 0:
                address = resume.get("contact_details", {}).get("address", "")
                location_match = False

                if address and address != "N/A":
                    for location in locations:
                        if search_city_in_address(address, location):
                            location_match = True
                            match_score += 5  # Bonus for location match
                            break

                if not location_match:
                    continue  # Skip if no location match

            # Add to filtered results with score
            formatted_resume = format_resume(resume)

            # Remove the total_resume_text field if present
            if "total_resume_text" in formatted_resume:
                del formatted_resume["total_resume_text"]

            formatted_resume["match_score"] = match_score
            filtered_results.append(formatted_resume)

        # Sort by combined score
        sorted_results = sorted(
            filtered_results, key=lambda x: x.get("match_score", 0), reverse=True
        )

        # Return top results up to limit
        return sorted_results[:limit]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


def search_city_in_address(address: str, search_city: str) -> bool:
    """
    Check if a city name exists in the address string.

    Args:
        address: The full address string
        search_city: The city name to search for

    Returns:
        True if the city is found in the address, False otherwise
    """
    if not address or not search_city or address == "N/A":
        return False

    # Normalize both strings for comparison
    address_lower = address.lower()
    search_city_lower = search_city.lower()

    # Simple check if the city name exists in the address
    return search_city_lower in address_lower


def parse_experience_to_months(experience_str: str) -> int:
    """
    Parse experience string in format 'X years Y months' or similar to total months.
    """
    if not experience_str:
        return 0

    years = 0
    months = 0

    # Extract years
    year_match = re.search(
        r"(\d+)\s*(?:year|years|yr|yrs)", experience_str, re.IGNORECASE
    )
    if year_match:
        years = int(year_match.group(1))

    # Extract months
    month_match = re.search(
        r"(\d+)\s*(?:month|months|mo|mos)", experience_str, re.IGNORECASE
    )
    if month_match:
        months = int(month_match.group(1))

    # If only a number is provided without units, assume it's years
    if not year_match and not month_match:
        num_match = re.search(r"(\d+)", experience_str)
        if num_match:
            years = int(num_match.group(1))

    return (years * 12) + months
