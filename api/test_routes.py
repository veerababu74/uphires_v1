import re
from fastapi import APIRouter, Body, HTTPException
from typing import List, Dict, Any, Optional
from database.client import get_collection
from core.helpers import format_resume
from core.vectorizer import Vectorizer

resumes_collection = get_collection()
vectorizer = Vectorizer()

router = APIRouter(
    prefix="/test",
    tags=["test_routes"],
)


@router.post("/semantic/", response_model=List[Dict[str, Any]])
async def semantic_resume_search(
    query: str = Body(..., description="Free text search query for the entire resume"),
    num_results: int = Body(10, description="Number of results to return"),
    min_experience: Optional[str] = Body(
        None, description="Minimum experience in format 'X years Y months'"
    ),
    min_education: Optional[str] = Body(
        None,
        description="Minimum education level (e.g., '10th', 'Diploma', 'Degree', 'BTech')",
    ),
    locations: Optional[List[str]] = Body(
        None, description="List of city names to search in address"
    ),
):
    """
    Perform semantic vector search on the entire content of resumes with optional filters.
    This provides more accurate results for natural language queries.
    """
    try:
        # Generate embedding for search query
        query_embedding = vectorizer.generate_embedding(query)

        # First perform vector search to get initial candidates
        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "total_resume_vector",
                        "k": num_results * 3,  # Get more initial results for filtering
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

        initial_results = list(resumes_collection.aggregate(pipeline))

        # Apply post-filters if specified
        filtered_results = []

        # Parse min_experience to months if provided
        min_experience_months = 0
        if min_experience:
            min_experience_months = parse_experience_to_months(min_experience)

        for resume in initial_results:
            should_include = True

            # Filter by minimum experience if specified
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
                    should_include = False

            # Filter by minimum education if specified
            if should_include and min_education:
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
                    # Get highest education level from resume
                    highest_level = 0
                    for edu in resume.get("education", []):
                        degree = edu.get("degree", "").lower()
                        for level_name, level_value in education_levels.items():
                            if level_name in degree:
                                highest_level = max(highest_level, level_value)

                    if highest_level < min_level:
                        should_include = False

            # Filter by locations if specified
            if should_include and locations and len(locations) > 0:
                address = resume.get("contact_details", {}).get("address", "")
                if address and address != "N/A":
                    location_match = False
                    for location in locations:
                        if search_city_in_address(address, location):
                            location_match = True
                            break

                    if not location_match:
                        should_include = False

            # Add resume to filtered results if it passed all filters
            if should_include:
                # Format the resume and remove unwanted fields
                formatted_resume = format_resume(resume)
                if "total_resume_text" in formatted_resume:
                    del formatted_resume["total_resume_text"]

                filtered_results.append(formatted_resume)

                # Break if we have enough results
                if len(filtered_results) >= num_results:
                    break

        return filtered_results
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Semantic resume search failed: {str(e)}"
        )


def search_city_in_address(address: str, search_city: str) -> bool:
    """
    Check if a city name exists in the address string.
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


import re
import numpy as np
from fastapi import APIRouter, Body, HTTPException
from typing import List, Dict, Any, Optional
from database.client import get_collection
from core.helpers import format_resume
from core.vectorizer import Vectorizer

# Get MongoDB collection
resumes_collection = get_collection()
enhanced_search_router = APIRouter(prefix="/ai", tags=["enhanced ai vector search"])
vectorizer = Vectorizer()


@enhanced_search_router.post("/search", response_model=List[Dict[str, Any]])
async def enhanced_vector_search(
    query: str = Body(..., description="Search query text"),
    limit: int = Body(10, description="Number of results to return"),
    min_score: float = Body(0.0, description="Minimum score threshold (0-1)"),
    strict_experience_filter: bool = Body(
        False, description="Extract and enforce experience requirements"
    ),
):
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")

        # Step 1: Process query
        min_experience_months, extracted_experience = extract_experience_requirement(
            query
        )
        clean_query = clean_query_text(query)

        # Step 2: Generate embedding
        query_vector = vectorizer.generate_embedding(clean_query)

        # Step 3: Search pipeline - Split into vector and text searches
        search_limit = limit * 3

        # Vector search pipeline
        vector_pipeline = [
            {
                "$search": {
                    "knnBeta": {
                        "vector": query_vector,
                        "path": "total_resume_vector",
                        "k": search_limit,
                    }
                }
            },
            {"$project": {"score": {"$meta": "searchScore"}, "document": "$$ROOT"}},
            {"$limit": search_limit},
        ]

        # Text search pipeline
        text_pipeline = [
            {"$search": {"text": {"query": clean_query, "path": {"wildcard": "*"}}}},
            {"$project": {"score": {"$meta": "searchScore"}, "document": "$$ROOT"}},
            {"$limit": search_limit},
        ]

        # Execute both searches
        vector_results = list(resumes_collection.aggregate(vector_pipeline))
        text_results = list(resumes_collection.aggregate(text_pipeline))

        # Combine and deduplicate results
        seen_ids = set()
        processed_results = []

        # Process vector results first
        for result in vector_results:
            if len(processed_results) >= limit:
                break

            resume = result["document"]
            resume_id = str(resume.get("_id"))

            if resume_id in seen_ids:
                continue

            seen_ids.add(resume_id)

            # Apply experience filter if needed
            if strict_experience_filter and min_experience_months > 0:
                total_exp = calculate_total_experience(resume)
                if total_exp < min_experience_months:
                    continue

            # Calculate scores
            relevance_score = calculate_relevance_score(resume, clean_query)
            match_score = result["score"]

            # Format resume
            formatted = format_resume(resume)
            formatted["match_score"] = match_score
            formatted["relevance_score"] = relevance_score
            formatted["final_score"] = (match_score * 0.7) + (relevance_score * 0.3)

            if formatted["final_score"] >= min_score:
                processed_results.append(formatted)

        # Process text results
        for result in text_results:
            if len(processed_results) >= limit:
                break

            resume = result["document"]
            resume_id = str(resume.get("_id"))

            if resume_id in seen_ids:
                continue

            seen_ids.add(resume_id)

            # Apply experience filter if needed
            if strict_experience_filter and min_experience_months > 0:
                total_exp = calculate_total_experience(resume)
                if total_exp < min_experience_months:
                    continue

            # Calculate scores
            relevance_score = calculate_relevance_score(resume, clean_query)
            match_score = result["score"]

            # Format resume
            formatted = format_resume(resume)
            formatted["match_score"] = match_score
            formatted["relevance_score"] = relevance_score
            formatted["final_score"] = (match_score * 0.7) + (relevance_score * 0.3)

            if formatted["final_score"] >= min_score:
                processed_results.append(formatted)

        # Sort by final score
        processed_results.sort(key=lambda x: x["final_score"], reverse=True)

        return processed_results[:limit]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Helper functions


def extract_experience_requirement(query: str) -> tuple:
    """
    Extract experience requirements from query string.

    Returns:
        tuple: (experience_months, extracted_text)
    """
    # Extract experience patterns like "5 years", "5+ years", "minimum 5 years", etc.
    exp_patterns = [
        r"(\d+)(?:\+)?\s*(?:year|years|yr|yrs)",  # 5 years, 5+ years
        r"(?:minimum|min|at least|above|more than|over)\s+(\d+)\s*(?:year|years|yr|yrs)",  # minimum 5 years
        r"(\d+)\s*(?:year|years|yr|yrs)(?:\+|\s+(?:above|or more|and above|experience|or greater))",  # 5 years+
    ]

    for pattern in exp_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            years = int(match.group(1))
            min_experience_months = years * 12
            extracted_experience = f"{years} years"
            return min_experience_months, extracted_experience

    # Check for explicit months
    month_patterns = [
        r"(\d+)(?:\+)?\s*(?:month|months|mo|mos)",  # 24 months, 24+ months
        r"(?:minimum|min|at least|above|more than|over)\s+(\d+)\s*(?:month|months|mo|mos)",  # minimum 24 months
    ]

    for pattern in month_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            months = int(match.group(1))
            return months, f"{months} months"

    return 0, None


def clean_query_text(query: str) -> str:
    """
    Clean query text by removing experience requirements and other non-essential terms.
    """
    # Remove experience requirements
    for term in [
        "years",
        "year",
        "yr",
        "yrs",
        "months",
        "month",
        "mo",
        "mos",
        "minimum",
        "min",
        "at least",
        "above",
        "more than",
        "over",
        "or more",
        "and above",
        "experience",
        "or greater",
        "+",
    ]:
        query = re.sub(r"\b" + re.escape(term) + r"\b", "", query)

    # Remove numbers
    query = re.sub(r"\b\d+\b", "", query)

    # Clean up extra spaces
    query = re.sub(r"\s+", " ", query).strip()

    return query


def calculate_total_experience(resume: Dict[str, Any]) -> int:
    """
    Calculate total experience months from a resume document.
    This function is flexible and attempts to find experience
    information in various possible schema layouts.
    """
    # Try to find total experience in a dedicated field
    if "total_experience" in resume:
        total_exp = resume.get("total_experience")
        if isinstance(total_exp, str):
            return parse_experience_to_months(total_exp)
        elif isinstance(total_exp, (int, float)):
            return int(total_exp)

    # Try to find experience in an "experience" array field
    total_months = 0

    # Look for an experience array
    experience_field = None
    for field in resume:
        if isinstance(resume[field], list) and field in [
            "experience",
            "work_experience",
            "employment",
        ]:
            experience_field = field
            break

    if experience_field:
        for exp in resume[experience_field]:
            if isinstance(exp, dict):
                # Try various possible field names for duration
                for duration_field in ["duration", "period", "length", "time"]:
                    if duration_field in exp:
                        duration = exp[duration_field]
                        if isinstance(duration, str):
                            total_months += parse_experience_to_months(duration)
                            break

    return total_months


def parse_experience_to_months(experience_str: str) -> int:
    """
    Parse experience string (e.g., "2 years 6 months", "3 years", "8 months") to total months.

    Args:
        experience_str (str): Experience duration string

    Returns:
        int: Total duration in months
    """
    if not isinstance(experience_str, str):
        return 0

    # Convert to lowercase for consistent matching
    exp_lower = experience_str.lower()

    # Pattern for years and months
    years_pattern = re.search(r"(\d+)(?:\s*)(year|years|yr|yrs)", exp_lower)
    months_pattern = re.search(r"(\d+)(?:\s*)(month|months|mo|mos)", exp_lower)

    total_months = 0

    # Add years (converted to months)
    if years_pattern:
        years = int(years_pattern.group(1))
        total_months += years * 12

    # Add months
    if months_pattern:
        months = int(months_pattern.group(1))
        total_months += months

    return total_months


def calculate_relevance_score(resume: Dict[str, Any], query: str) -> float:
    """
    Calculate a relevance score for a resume given the query.
    This function tries to be schema-agnostic and works with
    various possible resume structures.
    """
    query_terms = set(query.lower().split())

    # Initialize score
    score = 0.0
    term_matches = 0

    # Check for skills (flexible approach)
    skill_fields = ["skills", "skill_set", "technologies", "expertise"]
    for field in skill_fields:
        if field in resume:
            skills = resume[field]
            if isinstance(skills, list):
                # If skills is a list of strings
                if all(isinstance(s, str) for s in skills):
                    skill_text = " ".join(skills).lower()
                    for term in query_terms:
                        if term in skill_text:
                            term_matches += 1
                # If skills is a list of objects
                elif all(isinstance(s, dict) for s in skills):
                    for skill_obj in skills:
                        for _, value in skill_obj.items():
                            if isinstance(value, str) and any(
                                term in value.lower() for term in query_terms
                            ):
                                term_matches += 1

    # Check for job titles and descriptions in experience
    experience_fields = ["experience", "work_experience", "employment", "jobs"]
    for field in experience_fields:
        if field in resume and isinstance(resume[field], list):
            for exp in resume[field]:
                if isinstance(exp, dict):
                    # Check various possible field names for job title and description
                    for title_field in ["title", "position", "role", "job_title"]:
                        if title_field in exp and isinstance(exp[title_field], str):
                            title_text = exp[title_field].lower()
                            for term in query_terms:
                                if term in title_text:
                                    # Job title matches are highly relevant
                                    term_matches += 2

                    for desc_field in [
                        "description",
                        "responsibilities",
                        "summary",
                        "details",
                    ]:
                        if desc_field in exp and isinstance(exp[desc_field], str):
                            desc_text = exp[desc_field].lower()
                            for term in query_terms:
                                if term in desc_text:
                                    term_matches += 1

    # Consider education if present
    education_fields = ["education", "qualifications", "degrees"]
    for field in education_fields:
        if field in resume:
            education = resume[field]
            if isinstance(education, list):
                for edu in education:
                    if isinstance(edu, dict):
                        for edu_field in ["degree", "major", "field", "qualification"]:
                            if edu_field in edu and isinstance(edu[edu_field], str):
                                edu_text = edu[edu_field].lower()
                                for term in query_terms:
                                    if term in edu_text:
                                        term_matches += 1

    # Calculate final score - normalize by query terms
    if query_terms:
        return min(1.0, term_matches / (len(query_terms) * 2))
    return 0.0
