import re
from fastapi import APIRouter, Body, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from mangodatabase.client import get_collection
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
    min_education: Optional[List[str]] = Field(
        default=None,
        description="List of minimum education qualifications",
        example=["BTech", "BSc"],
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
    min_salary: Optional[float] = Field(
        default=None,
        description="Minimum expected salary",
        example=500000.0,
    )
    max_salary: Optional[float] = Field(
        default=None,
        description="Maximum expected salary",
        example=1500000.0,
    )
    limit: int = Field(
        default=10, ge=1, le=50, description="Maximum number of results", example=10
    )


router = APIRouter(
    prefix="/manualsearch_old",
    tags=["Manual Resume Search old version"],
)


@router.post(
    "/",
    response_model=List[Dict[str, Any]],
    summary="Advanced Resume Search",
    description="""
    Search for resumes using multiple criteria with intelligent ranking.

    **Search Criteria:**
    - Experience Titles (Required): Job titles to match (e.g., ["Software Engineer", "Developer"])
    - Skills (Optional): Technical skills to match (includes both 'skills' and 'may_also_known_skills')
    - Minimum Education (Optional): Education level (e.g., "BTech", "MCA", "MBA")
    - Minimum Experience (Optional): Required experience (e.g., "2 years 6 months")
    - Maximum Experience (Optional): Maximum experience limit (e.g., "5 years")
    - Locations (Optional): Preferred cities (matches both current_city and looking_for_jobs_in)
    - Min Salary (Optional): Minimum expected salary filter
    - Max Salary (Optional): Maximum expected salary filter (budget limit)
    
    **Salary Filtering:**
    - Uses expected_salary if available, falls back to current_salary
    - Candidates without salary information are excluded when salary filters are applied
    - Salary values should be in the same currency as stored in database
    
    **Data Structure:**
    - Contact Details: name, email, phone, current_city, looking_for_jobs_in, pan_card, etc.
    - Academic Details: education, college, pass_year
    - Experience: company, title, from_date, to (optional)
    - Skills: skills array and may_also_known_skills array
    - Financial: current_salary, expected_salary, currency, pay_duration
    
    **Education Levels Hierarchy:**
    1. 10th/SSC
    2. 12th/HSC/Intermediate
    3. Diploma/ITI
    4. Bachelor's (BTech/BE/BSc/BA/BCom)
    5. Master's (MTech/ME/MSc/MA/MCom/MBA)
    6. PhD/Doctorate
    
    **Scoring System:**
    - Experience Title Match: 10 points per match
    - Skills Match: Up to 10 points based on percentage match (includes may_also_known_skills)
    - Education Level Match: Points based on level (1-6)
    - Experience Duration Match: 5-10 points
    - Location Match: 5 points (current_city), 3 points (looking_for_jobs_in)
    - Salary Range Match: 5-8 points based on match type
    
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
                            "user_id": "user123",
                            "username": "johndoe",
                            "contact_details": {
                                "name": "John Doe",
                                "email": "john@example.com",
                                "phone": "+91-1234567890",
                                "current_city": "Mumbai",
                                "looking_for_jobs_in": ["Mumbai", "Pune"],
                                "pan_card": "ABCDE1234F",
                                "gender": "Male",
                                "age": 28,
                            },
                            "academic_details": [
                                {
                                    "education": "BTech in Computer Science",
                                    "college": "Mumbai University",
                                    "pass_year": 2020,
                                }
                            ],
                            "experience": [
                                {
                                    "company": "Tech Corp",
                                    "title": "Senior Software Engineer",
                                    "from_date": "2021-01",
                                    "to": "2023-12",
                                }
                            ],
                            "skills": ["Python", "React", "AWS"],
                            "may_also_known_skills": ["Docker", "Kubernetes"],
                            "total_experience": "2 years 6 months",  # Changed from 2.5 to string
                            "expected_salary": 1500000.0,
                            "current_salary": 1200000.0,
                            "notice_period": "30 days",
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
            )  # Step 1: Build the base query for experience titles
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

            match_score += (
                exp_title_matches * 10
            )  # Weight for title matches            # Filter by skills if provided
            if search_params.skills and len(search_params.skills) > 0:
                resume_skills = [skill.lower() for skill in resume.get("skills", [])]
                # Also check may_also_known_skills
                may_also_known_skills = [
                    skill.lower() for skill in resume.get("may_also_known_skills", [])
                ]
                all_resume_skills = resume_skills + may_also_known_skills

                skill_matches = sum(
                    1
                    for skill in search_params.skills
                    if skill.lower() in all_resume_skills
                )

                # Add skill match score
                if skill_matches > 0:
                    match_score += (skill_matches / len(search_params.skills)) * 10
                else:
                    # Skip this resume if no skills match
                    continue  # Filter by minimum education if provided
            if search_params.min_education and len(search_params.min_education) > 0:
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

                # Get minimum required levels from search parameters
                min_levels = [
                    education_levels.get(edu.lower(), 0)
                    for edu in search_params.min_education
                ]
                min_levels = [level for level in min_levels if level > 0]

                if min_levels:  # Only apply filter if valid education levels found
                    # Extract highest education level from resume
                    highest_level = 0
                    for edu in resume.get("academic_details", []):
                        education = edu.get("education", "").lower()
                        # Check each word in the education against the education levels
                        for level_name, level_value in education_levels.items():
                            if level_name in education:
                                highest_level = max(highest_level, level_value)

                    # Check if highest level meets any of the minimum requirements
                    if not any(highest_level >= min_level for min_level in min_levels):
                        continue  # Skip if education requirements not met

                    match_score += highest_level  # Add score based on education level            # Replace the existing experience filtering block with this:
            if min_experience_months > 0 or max_experience_months < float("inf"):
                # Try to parse total_experience field
                total_exp = resume.get("total_experience", "")
                resume_exp_months = 0

                if total_exp and total_exp != "N/A":
                    resume_exp_months = parse_experience_to_months(str(total_exp))
                else:
                    # Calculate from individual experiences if total not available
                    for exp in resume.get("experience", []):
                        exp_months = calculate_experience_from_dates(
                            exp.get("from_date", ""), exp.get("to", "")
                        )
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
                    match_score += 5  # Close to required minimum experience# Filter by locations if provided
            if search_params.locations and len(search_params.locations) > 0:
                # Check both current_city and looking_for_jobs_in
                current_city = resume.get("contact_details", {}).get("current_city", "")
                looking_for_jobs_in = resume.get("contact_details", {}).get(
                    "looking_for_jobs_in", []
                )

                location_match = False  # Check current city
                if current_city and current_city != "N/A":
                    for location in search_params.locations:
                        if search_city_in_location(current_city, location):
                            location_match = True
                            match_score += 5  # Bonus for current city match
                            break

                # Check preferred job locations
                if not location_match and looking_for_jobs_in:
                    for job_location in looking_for_jobs_in:
                        if job_location and job_location != "N/A":
                            for location in search_params.locations:
                                if search_city_in_location(job_location, location):
                                    location_match = True
                                    match_score += (
                                        3  # Slightly lower bonus for preferred location
                                    )
                                    break
                        if location_match:
                            break

                if not location_match and len(search_params.locations) > 0:
                    continue  # Skip if no location match and locations were specified

            # Filter by salary range if provided
            if (
                search_params.min_salary is not None
                or search_params.max_salary is not None
            ):
                expected_salary = resume.get("expected_salary", 0)
                current_salary = resume.get("current_salary", 0)

                # Use expected_salary if available, otherwise use current_salary
                candidate_salary = (
                    expected_salary
                    if expected_salary and expected_salary > 0
                    else current_salary
                )

                # Skip if no salary information available and salary filters are specified
                if not candidate_salary or candidate_salary <= 0:
                    if (
                        search_params.min_salary is not None
                        or search_params.max_salary is not None
                    ):
                        continue

                # Check minimum salary requirement
                if (
                    search_params.min_salary is not None
                    and candidate_salary < search_params.min_salary
                ):
                    continue  # Skip if salary is below minimum requirement

                # Check maximum salary requirement
                if (
                    search_params.max_salary is not None
                    and candidate_salary > search_params.max_salary
                ):
                    continue  # Skip if salary is above maximum budget

                # Add score bonus for salary match
                if (
                    search_params.min_salary is not None
                    and search_params.max_salary is not None
                ):
                    # Perfect range match
                    if (
                        search_params.min_salary
                        <= candidate_salary
                        <= search_params.max_salary
                    ):
                        match_score += 8
                elif search_params.min_salary is not None:
                    # Above minimum requirement
                    if candidate_salary >= search_params.min_salary:
                        match_score += 5
                elif search_params.max_salary is not None:
                    # Within budget
                    if candidate_salary <= search_params.max_salary:
                        match_score += 5

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


def search_city_in_location(location: str, search_city: str) -> bool:
    """
    Check if a city name exists in the location string.

    Args:
        location: The location string (current_city or job location)
        search_city: The city name to search for

    Returns:
        True if the city is found in the location, False otherwise
    """
    if not location or not search_city or location == "N/A":
        return False

    # Normalize both strings for comparison
    location_lower = location.lower()
    search_city_lower = search_city.lower()

    # Simple check if the city name exists in the location
    return search_city_lower in location_lower


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


def calculate_experience_from_dates(from_date: str, to_date: str = None) -> int:
    """
    Calculate experience in months from date range.

    Args:
        from_date: Start date in 'YYYY-MM' format
        to_date: End date in 'YYYY-MM' format (None for current job)

    Returns:
        Experience duration in months
    """
    if not from_date:
        return 0

    try:
        # Parse from_date
        from_parts = from_date.split("-")
        if len(from_parts) != 2:
            return 0

        from_year = int(from_parts[0])
        from_month = int(from_parts[1])

        # Parse to_date or use current date
        if to_date and to_date.strip():
            to_parts = to_date.split("-")
            if len(to_parts) != 2:
                return 0
            to_year = int(to_parts[0])
            to_month = int(to_parts[1])
        else:
            # Use current date for ongoing jobs
            from datetime import datetime

            current_date = datetime.now()
            to_year = current_date.year
            to_month = current_date.month

        # Calculate months difference
        total_months = (to_year - from_year) * 12 + (to_month - from_month)

        # Return at least 1 month for any valid experience
        return max(1, total_months)

    except (ValueError, IndexError):
        # If date parsing fails, return 0
        return 0
