from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from pymongo.collection import Collection
from mangodatabase.operations import ResumeOperations
from embeddings.vectorizer import Vectorizer
from core.database import get_database
from core.vectorizer import get_vectorizer

router = APIRouter(prefix="/resumes", tags=["resumes crud"])


def get_resume_operations(
    db: Collection = Depends(get_database),
    vectorizer: Vectorizer = Depends(get_vectorizer),
) -> ResumeOperations:
    return ResumeOperations(db, vectorizer)


@router.post("/", response_model=Dict[str, str])
async def create_resume(
    resume_data: Dict[str, Any],
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Create a new resume with vector embeddings"""
    return operations.create_resume(resume_data)


@router.put("/{resume_id}", response_model=Dict[str, str])
async def update_resume(
    resume_id: str,
    resume_data: Dict[str, Any],
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Update an existing resume with vector embeddings"""
    return operations.update_resume(resume_id, resume_data)


@router.get("/{resume_id}", response_model=Dict[str, Any])
async def get_resume(
    resume_id: str, operations: ResumeOperations = Depends(get_resume_operations)
):
    """Get a resume by ID"""
    return operations.get_resume(resume_id)


@router.delete("/{resume_id}", response_model=Dict[str, str])
async def delete_resume(
    resume_id: str, operations: ResumeOperations = Depends(get_resume_operations)
):
    """Delete a resume by ID"""
    return operations.delete_resume(resume_id)


@router.get("/", response_model=List[Dict[str, Any]])
async def list_resumes(
    skip: int = 0,
    limit: int = 10,
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """List all resumes with pagination"""
    return operations.list_resumes(skip, limit)


@router.post("/update-embeddings", response_model=Dict[str, str])
async def update_all_embeddings(
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Update vector embeddings for all resumes"""
    return operations.update_all_vector_embeddings()
