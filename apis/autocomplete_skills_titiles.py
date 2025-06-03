from fastapi import APIRouter, Query, Body, Depends, HTTPException
from mangodatabase.client import get_collection
from embeddings.vectorizer import Vectorizer
from core.helpers import format_resume
from typing import List, Dict, Any
import pymongo
import re

router = APIRouter(
    prefix="/autocomplete",
    tags=["Job Search Autocomplete"],
)

collection = get_collection()
vectorizer = Vectorizer()


def process_results(results, key):
    """
    Process the results by converting to lowercase, stripping whitespace,
    and removing duplicates while maintaining order.
    Also filters out empty or blank strings.
    """
    processed = []
    seen = set()
    for result in results:
        value = result.get(key)
        if not isinstance(value, str):
            continue  # Skip invalid or None values
        stripped = value.strip()
        if not stripped:  # Skip empty or whitespace-only strings
            continue
        lower_value = stripped.lower()
        if lower_value not in seen:
            seen.add(lower_value)
            processed.append(value)  # Preserve original casing
    return processed


def extract_skills(raw_data: str) -> List[str]:
    if not raw_data or not isinstance(raw_data, str):
        return []

    cleaned = re.sub(r".*?:", "", raw_data)
    parenthetical_content = re.findall(r"\((.*?)\)", cleaned)
    cleaned = re.sub(r"\(.*?\)", ",", cleaned)
    skills = re.split(r"[,/&]|\band\b", cleaned)
    for content in parenthetical_content:
        skills.extend(re.split(r"[,\s]+", content))

    processed_skills = []
    for skill in skills:
        skill = re.sub(r"[^\w\s-]", "", skill).strip().lower()
        if skill and skill not in {"others", "and", "in", "of"}:
            processed_skills.append(skill)
    return processed_skills


@router.get(
    "/job_titles/",
    response_model=List[str],
    summary="Autocomplete Job Titles",
    description="""
    Get autocomplete suggestions for job titles based on input prefix.
    Uses both exact matching and semantic search for better results.
    **Parameters:**
    - prefix: Text to search for in job titles (e.g., "software eng")
    - limit: Maximum number of suggestions to return
    **Returns:**
    List of matching job titles sorted by relevance
    **examples Usage:**
    - prefix="soft" might return ["Software Engineer", "Software Developer", "Software Architect"]
    - prefix="data" might return ["Data Scientist", "Data Engineer", "Data Analyst"]
    """,
    responses={
        200: {
            "description": "Successful job title suggestions",
            "content": {
                "application/json": {
                    "example": [
                        "software engineer",
                        "software developer",
                        "senior software engineer",
                        "software architect",
                        "software team lead",
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix cannot be empty"}
                }
            },
        },
    },
)
async def autocomplete_titles(
    prefix: str = Query(
        ...,
        description="Job title prefix to search for",
        min_length=2,
        example="software eng",
    ),
    limit: int = Query(
        default=10, description="Maximum number of suggestions", ge=1, le=50, example=5
    ),
):
    try:
        pipeline = [
            {"$unwind": "$experience"},
            {
                "$match": {
                    "experience.title": {"$regex": f".*{prefix}.*", "$options": "i"}
                }
            },
            {"$group": {"_id": "$experience.title"}},
            {"$limit": limit},
            {"$project": {"title": "$_id", "_id": 0}},
        ]
        results = list(collection.aggregate(pipeline))
        titles = process_results(results, "title")

        if len(titles) < limit:
            query_embedding = vectorizer.generate_embedding(prefix)
            semantic_pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "experience_text_vector",
                            "k": limit,
                        },
                    }
                },
                {"$unwind": "$experience"},
                {"$project": {"title": "$experience.title"}},
                {"$limit": limit},
            ]
            semantic_results = list(collection.aggregate(semantic_pipeline))
            semantic_titles = process_results(semantic_results, "title")
            titles.extend(
                [
                    title
                    for title in semantic_titles
                    if title.lower().strip()
                    not in map(str.lower, map(str.strip, titles))
                ]
            )

        return titles[:limit]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Title autocomplete failed: {str(e)}"
        )


@router.get(
    "/job_skillsv1/",
    response_model=List[str],
    summary="Autocomplete Technical Skills",
    description="""
    Get autocomplete suggestions for technical skills based on input prefix.
    Uses both exact matching and semantic search for better results.
    **Parameters:**
    - prefix: Text to search for in skills (e.g., "py" for Python)
    - limit: Maximum number of suggestions to return
    **Returns:**
    List of matching skills sorted by relevance
    **examples Usage:**
    - prefix="py" might return ["python", "pytorch", "pyqt"]
    - prefix="java" might return ["javascript", "java", "java spring"]
    """,
    responses={
        200: {
            "description": "Successful skill suggestions",
            "content": {
                "application/json": {
                    "example": [
                        "python",
                        "pytorch",
                        "python django",
                        "python flask",
                        "python scripting",
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix cannot be empty"}
                }
            },
        },
    },
)
async def autocomplete_skills(
    prefix: str = Query(
        ..., description="Skill prefix to search for", min_length=2, example="py"
    ),
    limit: int = Query(
        default=10, description="Maximum number of suggestions", ge=1, le=50, example=5
    ),
):
    try:
        # First, let's try a broader search to see if we have any skills data
        pipeline = [
            {"$unwind": "$skills"},
            {"$match": {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}}},
            {"$group": {"_id": "$skills"}},
            {"$limit": limit * 5},  # Increase limit to account for splitting
            {"$project": {"raw_skill": "$_id", "_id": 0}},
        ]
        raw_results = list(collection.aggregate(pipeline))

        # If no exact matches, try a different field or broader search
        if not raw_results:
            # Try searching in different skill fields or use a more generic approach
            alternative_pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                            {
                                "technical_skills": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                            {
                                "experience.skills": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                        ]
                    }
                },
                {"$limit": limit * 2},
            ]
            alternative_results = list(collection.aggregate(alternative_pipeline))

            # Extract skills from various fields
            all_skills = []
            for doc in alternative_results:
                if doc.get("skills"):
                    if isinstance(doc["skills"], list):
                        all_skills.extend(doc["skills"])
                    else:
                        all_skills.append(doc["skills"])
                if doc.get("technical_skills"):
                    if isinstance(doc["technical_skills"], list):
                        all_skills.extend(doc["technical_skills"])
                    else:
                        all_skills.append(doc["technical_skills"])

            # Filter skills that match the prefix
            matching_skills = [
                skill.strip()
                for skill in all_skills
                if isinstance(skill, str) and prefix.lower() in skill.lower()
            ]

            # Remove duplicates while preserving order
            unique_skills = []
            seen = set()
            for skill in matching_skills:
                skill_lower = skill.lower()
                if skill_lower not in seen:
                    seen.add(skill_lower)
                    unique_skills.append(skill)

            return unique_skills[:limit]

        raw_skills = [
            result["raw_skill"] for result in raw_results if result.get("raw_skill")
        ]

        extracted_skills = [
            skill for raw_skill in raw_skills for skill in extract_skills(raw_skill)
        ]

        unique_skills = list({skill for skill in extracted_skills})
        sorted_skills = sorted(
            unique_skills, key=lambda s: (not s.startswith(prefix.lower()), s)
        )

        return sorted_skills[:limit]
    except Exception as e:
        # Return a more detailed error for debugging
        raise HTTPException(
            status_code=500, detail=f"Skills autocomplete failed: {str(e)}"
        )


# New combined route
@router.get(
    "/jobs_and_skills/",
    response_model=Dict[str, List[str]],
    summary="Autocomplete Job Titles and Skills",
    description="""
    Get autocomplete suggestions for both job titles and technical skills based on input prefix.
    Combines results from exact matching and semantic search.
    **Parameters:**
    - prefix: Text to search for in both job titles and skills (e.g., "py" for Python-related jobs and skills)
    - limit: Maximum number of suggestions for each category
    **Returns:**
    Object containing two arrays: one for matching job titles and one for matching skills
    **examples Response:**
    {
      "titles": ["Python Developer", "Senior Python Engineer", "Python Team Lead"],
      "skills": ["Python", "Python Django", "Python Flask"]
    }
    """,
    responses={
        200: {
            "description": "Successful job title and skill suggestions",
            "content": {
                "application/json": {
                    "example": {
                        "titles": [
                            "Python Developer",
                            "Senior Python Engineer",
                            "Python Team Lead",
                        ],
                        "skills": [
                            "Python",
                            "Python Django",
                            "Python Flask",
                        ],
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix cannot be empty"}
                }
            },
        },
    },
)
async def autocomplete_jobs_and_skills(
    prefix: str = Query(
        ...,
        description="Prefix to search in titles and skills",
        min_length=2,
        example="py",
    ),
    limit: int = Query(
        default=5, description="Maximum number of suggestions per category", ge=1, le=50
    ),
):
    try:
        # --- Fetch and process job titles ---
        title_pipeline = [
            {"$unwind": "$experience"},
            {
                "$match": {
                    "experience.title": {"$regex": f".*{prefix}.*", "$options": "i"}
                }
            },
            {"$group": {"_id": "$experience.title"}},
            {"$limit": limit},
            {"$project": {"title": "$_id", "_id": 0}},
        ]
        title_results = list(collection.aggregate(title_pipeline))
        titles = process_results(title_results, "title")

        if len(titles) < limit:
            query_embedding = vectorizer.generate_embedding(prefix)
            title_semantic_pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "experience_text_vector",
                            "k": limit,
                        },
                    }
                },
                {"$unwind": "$experience"},
                {"$project": {"title": "$experience.title"}},
                {"$limit": limit},
            ]
            semantic_title_results = list(collection.aggregate(title_semantic_pipeline))
            semantic_titles = process_results(semantic_title_results, "title")
            titles.extend(
                [
                    title
                    for title in semantic_titles
                    if title.lower().strip()
                    not in map(str.lower, map(str.strip, titles))
                ]
            )

        # --- Fetch and process skills ---
        skill_pipeline = [
            {"$unwind": "$skills"},
            {"$match": {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}}},
            {"$group": {"_id": "$skills"}},
            {"$limit": limit * 5},
            {"$project": {"raw_skill": "$_id", "_id": 0}},
        ]
        raw_skill_results = list(collection.aggregate(skill_pipeline))
        raw_skills = [
            result["raw_skill"]
            for result in raw_skill_results
            if result.get("raw_skill")
        ]
        extracted_skills = [
            skill for raw_skill in raw_skills for skill in extract_skills(raw_skill)
        ]
        unique_skills = list(set(extracted_skills))
        sorted_skills = sorted(
            unique_skills, key=lambda s: (not s.startswith(prefix.lower()), s)
        )

        return {
            "titles": titles[:limit],
            "skills": sorted_skills[:limit],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Combined autocomplete failed: {str(e)}"
        )


# @router.get(
#     "/debug/sample_data",
#     summary="Debug: Sample Data Structure",
#     description="Returns a sample document to understand the data structure",
# )
# async def get_sample_data():
#     try:
#         # Get a sample document to understand the structure
#         sample = collection.find_one({}, {"_id": 0})
#         if sample:
#             # Show keys and sample values
#             structure = {}
#             for key, value in sample.items():
#                 if isinstance(value, list) and value:
#                     structure[key] = (
#                         f"Array with {len(value)} items, sample: {value[0] if value else 'empty'}"
#                     )
#                 elif isinstance(value, dict):
#                     structure[key] = f"Object with keys: {list(value.keys())}"
#                 else:
#                     structure[key] = (
#                         f"Type: {type(value).__name__}, value: {str(value)[:100]}"
#                     )
#             return {"structure": structure, "sample_keys": list(sample.keys())}
#         else:
#             return {"message": "No documents found in collection"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


# @router.get(
#     "/debug/skills_fields",
#     summary="Debug: Available Skills Fields",
#     description="Check what skill-related fields exist in the database",
# )
# async def check_skills_fields():
#     try:
#         # Check for various skill-related fields
#         pipeline = [
#             {"$limit": 10},
#             {
#                 "$project": {
#                     "has_skills": {"$ifNull": ["$skills", "NOT_FOUND"]},
#                     "has_technical_skills": {
#                         "$ifNull": ["$technical_skills", "NOT_FOUND"]
#                     },
#                     "has_experience_skills": {
#                         "$ifNull": ["$experience.skills", "NOT_FOUND"]
#                     },
#                     "skills_type": {"$type": "$skills"},
#                     "skills_sample": {"$slice": ["$skills", 3]},
#                 }
#             },
#         ]
#         results = list(collection.aggregate(pipeline))
#         return {"sample_documents": results, "total_checked": len(results)}
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Skills fields check failed: {str(e)}"
#         )
