from mangodatabase.operations import SkillsTitlesOperations
from mangodatabase.client import get_skills_titles_collection
from fastapi import APIRouter, HTTPException, UploadFile, File

router = APIRouter(prefix="/skills_titles", tags=["Skills Titles"])

collection = get_skills_titles_collection()
skills_titles_operations = SkillsTitlesOperations(collection)


@router.get("/get_all_skills")
async def get_all_skills_titles():
    """Get all skills and titles from the database"""
    try:
        skills = skills_titles_operations.get_all_skills()

        return {
            "skills": skills,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching skills and titles: {str(e)}"
        )


@router.get("/get_all_titles")
async def get_all_titles():
    """Get all titles from the database"""
    try:
        titles = skills_titles_operations.get_all_titles()

        return {
            "titles": titles,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching skills and titles: {str(e)}"
        )
    """Get all titles from the database"""
