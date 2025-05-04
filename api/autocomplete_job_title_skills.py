# resume_api/api/autocomplete.py
from fastapi import APIRouter, Query, Body, Depends, HTTPException
from database.client import get_collection
from core.vectorizer import Vectorizer
from core.helpers import format_resume
from typing import List, Dict
import pymongo
import re

router = APIRouter(
    prefix="/autocomplete",
    tags=["Job Search Autocomplete"],
)

collection = get_collection()
vectorizer = Vectorizer()


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
    
    **Example Usage:**
    - prefix="soft" might return ["Software Engineer", "Software Developer", "Software Architect"]
    - prefix="data" might return ["Data Scientist", "Data Engineer", "Data Analyst"]
    """,
    responses={
        200: {
            "description": "Successful job title suggestions",
            "content": {
                "application/json": {
                    "example": [
                        "Software Engineer",
                        "Software Developer",
                        "Senior Software Engineer",
                        "Software Architect",
                        "Software Team Lead",
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
        titles = [doc["title"] for doc in results]

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
            semantic_titles = [
                doc["title"] for doc in semantic_results if doc["title"] not in titles
            ]
            titles.extend(semantic_titles[: limit - len(titles)])

        return titles

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Title autocomplete failed: {str(e)}"
        )


@router.get(
    "/job_skills/",
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
    
    **Example Usage:**
    - prefix="py" might return ["Python", "PyTorch", "PyQt"]
    - prefix="java" might return ["JavaScript", "Java", "Java Spring"]
    """,
    responses={
        200: {
            "description": "Successful skill suggestions",
            "content": {
                "application/json": {
                    "example": [
                        "Python",
                        "PyTorch",
                        "Python Django",
                        "Python Flask",
                        "Python Scripting",
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
        pipeline = [
            {"$unwind": "$skills"},
            {"$match": {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}}},
            {"$group": {"_id": "$skills"}},
            {"$limit": limit},
            {"$project": {"skill": "$_id", "_id": 0}},
        ]

        results = list(collection.aggregate(pipeline))
        skills = [doc["skill"] for doc in results]

        if len(skills) < limit:
            query_embedding = vectorizer.generate_embedding(prefix)

            semantic_pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "skills_vector",
                            "k": limit,
                        },
                    }
                },
                {"$unwind": "$skills"},
                {"$project": {"skill": "$skills"}},
                {"$limit": limit},
            ]

            semantic_results = list(collection.aggregate(semantic_pipeline))
            semantic_skills = [
                doc["skill"] for doc in semantic_results if doc["skill"] not in skills
            ]
            skills.extend(semantic_skills[: limit - len(skills)])

        return skills

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Skills autocomplete failed: {str(e)}"
        )


@router.get(
    "/search/skills/job_titles/",
    response_model=Dict[str, List[str]],
    summary="Combined Job Title and Skills Search",
    description="""
    Search for both job titles and skills simultaneously.
    Returns matching suggestions for both categories.
    
    **Parameters:**
    - prefix: Text to search for in both titles and skills
    - limit: Maximum number of suggestions per category
    
    **Returns:**
    Dictionary with two lists:
    - titles: Matching job titles
    - skills: Matching technical skills
    
    **Example Usage:**
    - prefix="python" returns both Python-related job titles and skills
    """,
    responses={
        200: {
            "description": "Successful combined search results",
            "content": {
                "application/json": {
                    "example": {
                        "titles": [
                            "Python Developer",
                            "Senior Python Engineer",
                            "Python Team Lead",
                        ],
                        "skills": ["Python", "Python Django", "Python Flask"],
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
async def autocomplete_search(
    prefix: str = Query(
        ...,
        description="Text to search in both titles and skills",
        min_length=2,
        example="python",
    ),
    limit: int = Query(
        default=10,
        description="Maximum suggestions per category",
        ge=1,
        le=50,
        example=5,
    ),
):
    try:
        # Get titles
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
        titles = [doc["title"] for doc in title_results]

        # Get skills
        skill_pipeline = [
            {"$unwind": "$skills"},
            {"$match": {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}}},
            {"$group": {"_id": "$skills"}},
            {"$limit": limit},
            {"$project": {"skill": "$_id", "_id": 0}},
        ]

        skill_results = list(collection.aggregate(skill_pipeline))
        skills = [doc["skill"] for doc in skill_results]

        # If results are less than limit, use semantic search
        if len(titles) < limit or len(skills) < limit:
            query_embedding = vectorizer.generate_embedding(prefix)

            # Semantic search for titles
            if len(titles) < limit:
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

                semantic_title_results = list(
                    collection.aggregate(title_semantic_pipeline)
                )
                semantic_titles = [
                    doc["title"]
                    for doc in semantic_title_results
                    if doc["title"] not in titles
                ]
                titles.extend(semantic_titles[: limit - len(titles)])

            # Semantic search for skills
            if len(skills) < limit:
                skill_semantic_pipeline = [
                    {
                        "$search": {
                            "index": "vector_search_index",
                            "knnBeta": {
                                "vector": query_embedding,
                                "path": "skills_vector",
                                "k": limit,
                            },
                        }
                    },
                    {"$unwind": "$skills"},
                    {"$project": {"skill": "$skills"}},
                    {"$limit": limit},
                ]

                semantic_skill_results = list(
                    collection.aggregate(skill_semantic_pipeline)
                )
                semantic_skills = [
                    doc["skill"]
                    for doc in semantic_skill_results
                    if doc["skill"] not in skills
                ]
                skills.extend(semantic_skills[: limit - len(skills)])

        return {"titles": titles, "skills": skills}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Combined autocomplete search failed: {str(e)}"
        )
