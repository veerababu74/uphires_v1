import re
from fastapi import APIRouter, Body, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from database.client import get_collection
from core.helpers import format_resume

resumes_collection = get_collection()


# Define request and response models for better documentation
class ManualSearchRequest(BaseModel):
    experience_titles: List[str] = Field(
        ...,
        description="List of job titles to search for",
        example=["Software Engineer", "Python Developer"],
    )
    skills: Optional[List[str]] = Field(
        default=None,
        description="Technical skills to match",
        example=["Python", "React", "AWS"],
    )
    min_education: Optional[str] = Field(
        default=None, description="Minimum education qualification", example="BTech"
    )
    min_experience: Optional[str] = Field(
        default=None,
        description="Minimum required experience",
        example="2 years 6 months",
    )
    max_experience: Optional[str] = Field(
        default=None,
        description="Maximum required experience",
        example="5 years",
    )
    locations: Optional[List[str]] = Field(
        default=None,
        description="Preferred locations",
        example=["Mumbai", "Pune", "Bangalore"],
    )
    limit: int = Field(
        default=10, ge=1, le=50, description="Maximum number of results", example=10
    )


router = APIRouter(
    prefix="/manualsearch",
    tags=["Manual Resume Search"],
)


@router.post(
    "/",
    response_model=List[Dict[str, Any]],
    summary="Advanced Resume Search",
    description="""
    Search for resumes using multiple criteria with intelligent ranking.

    **Search Criteria:**
    - Experience Titles (Required): Job titles to match (e.g., ["Software Engineer", "Developer"])
    - Skills (Optional): Technical skills to match (e.g., ["Python", "React"])
    - Minimum Education (Optional): Education level (e.g., "BTech", "MCA", "MBA")
    - Minimum Experience (Optional): Required experience (e.g., "2 years 6 months")
    - Locations (Optional): Preferred cities (e.g., ["Mumbai", "Bangalore"])
    
    **Education Levels Hierarchy:**
    1. 10th/SSC
    2. 12th/HSC/Intermediate
    3. Diploma/ITI
    4. Bachelor's (BTech/BE/BSc/BA/BCom)
    5. Master's (MTech/ME/MSc/MA/MCom/MBA)
    6. PhD/Doctorate
    
    **Scoring System:**
    - Experience Title Match: 10 points per match
    - Skills Match: Up to 10 points based on percentage match
    - Education Level Match: Points based on level (1-6)
    - Experience Duration Match: 5-10 points
    - Location Match: 5 points
    
    **Returns:**
    Sorted list of matching resumes with scores
    """,
    responses={
        200: {
            "description": "Successful search results",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "name": "John Doe",
                            "contact_details": {
                                "email": "john@example.com",
                                "phone": "+91-1234567890",
                                "address": "Mumbai, Maharashtra",
                            },
                            "education": [
                                {
                                    "degree": "BTech in Computer Science",
                                    "institution": "Mumbai University",
                                    "year": "2020",
                                }
                            ],
                            "experience": [
                                {
                                    "title": "Senior Software Engineer",
                                    "company": "Tech Corp",
                                    "duration": "2 years",
                                    "description": "Full stack development",
                                }
                            ],
                            "skills": ["Python", "React", "AWS"],
                            "total_experience": 2.5,
                            "match_score": 85.5,
                        }
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "At least one experience title is required"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "Manual search failed: Database error"}
                }
            },
        },
    },
)
async def manual_resume_search(search_params: ManualSearchRequest):
    try:
        if not search_params.experience_titles:
            raise HTTPException(
                status_code=400, detail="At least one experience title is required"
            )

        # Parse min and max experience strings to months
        min_experience_months = 0
        max_experience_months = float("inf")  # Default to infinity if not specified

        if search_params.min_experience:
            min_experience_months = parse_experience_to_months(
                search_params.min_experience
            )

        if search_params.max_experience:
            max_experience_months = parse_experience_to_months(
                search_params.max_experience
            )

        # Step 1: Build the base query for experience titles
        title_patterns = [
            re.compile(f".*{re.escape(title)}.*", re.IGNORECASE)
            for title in search_params.experience_titles
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
            if len(filtered_results) >= search_params.limit:
                break

            match_score = 0
            should_include = True

            # Calculate experience title match score
            exp_title_matches = 0
            for exp in resume.get("experience", []):
                exp_title = exp.get("title", "").lower()
                for title in search_params.experience_titles:
                    if title.lower() in exp_title:
                        exp_title_matches += 1
                        break

            # Only include if at least one experience title matches
            if exp_title_matches == 0:
                continue

            match_score += exp_title_matches * 10  # Weight for title matches

            # Filter by skills if provided
            if search_params.skills and len(search_params.skills) > 0:
                resume_skills = [skill.lower() for skill in resume.get("skills", [])]
                skill_matches = sum(
                    1
                    for skill in search_params.skills
                    if skill.lower() in resume_skills
                )

                # Add skill match score
                if skill_matches > 0:
                    match_score += (skill_matches / len(search_params.skills)) * 10
                else:
                    # Skip this resume if no skills match
                    continue

            # Filter by minimum education if provided
            if search_params.min_education:
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

                min_level = education_levels.get(search_params.min_education.lower(), 0)
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

            # Replace the existing experience filtering block with this:
            if min_experience_months > 0 or max_experience_months < float("inf"):
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

                # Check both min and max experience conditions
                if (
                    resume_exp_months < min_experience_months
                    or resume_exp_months > max_experience_months
                ):
                    continue  # Skip if experience requirements not met

                # Add score based on experience match
                if min_experience_months <= resume_exp_months <= max_experience_months:
                    match_score += 10  # Perfect range match
                elif resume_exp_months <= min_experience_months * 1.5:
                    match_score += 5  # Close to required minimum experience

            # Filter by locations if provided
            if search_params.locations and len(search_params.locations) > 0:
                address = resume.get("contact_details", {}).get("address", "")
                location_match = False

                if address and address != "N/A":
                    # Simple check if any of the requested locations exist in the address
                    location_match = False
                    for location in search_params.locations:
                        if search_city_in_address(address, location):
                            location_match = True
                            match_score += 5  # Bonus for location match
                            break

                if not location_match and len(search_params.locations) > 0:
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
        return sorted_results[: search_params.limit]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual search failed: {str(e)}")


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
