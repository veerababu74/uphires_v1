from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from pymongo.collection import Collection
from mangodatabase.operations import ResumeOperations
from embeddings.vectorizer import AddUserDataVectorizer
from core.database import get_database
from mangodatabase.client import get_collection

router = APIRouter(prefix="/resumes", tags=["resumes crud"])

# Global vectorizer instance
_vectorizer = None


def get_vectorizer() -> AddUserDataVectorizer:
    """Get or create the AddUserDataVectorizer instance"""
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = AddUserDataVectorizer()
    return _vectorizer


def get_resume_operations(
    db: Collection = Depends(get_database),
) -> ResumeOperations:
    vectorizer = get_vectorizer()
    return ResumeOperations(db, vectorizer)


@router.post("/", response_model=Dict[str, str])
async def create_resume(
    resume_data: Dict[str, Any],
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Create a new resume with vector embeddings"""
    try:
        if not resume_data:
            raise HTTPException(status_code=400, detail="Resume data cannot be empty")

        result = operations.create_resume(resume_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/{resume_id}", response_model=Dict[str, str])
async def update_resume(
    resume_id: str,
    resume_data: Dict[str, Any],
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Update an existing resume with vector embeddings"""
    try:
        if not resume_id.strip():
            raise HTTPException(status_code=400, detail="Resume ID cannot be empty")

        if not resume_data:
            raise HTTPException(status_code=400, detail="Resume data cannot be empty")

        result = operations.update_resume(resume_id, resume_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{resume_id}", response_model=Dict[str, Any])
async def get_resume(
    resume_id: str, operations: ResumeOperations = Depends(get_resume_operations)
):
    """Get a resume by ID"""
    try:
        if not resume_id.strip():
            raise HTTPException(status_code=400, detail="Resume ID cannot be empty")

        result = operations.get_resume(resume_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{resume_id}", response_model=Dict[str, str])
async def delete_resume(
    resume_id: str, operations: ResumeOperations = Depends(get_resume_operations)
):
    """Delete a resume by ID"""
    try:
        if not resume_id.strip():
            raise HTTPException(status_code=400, detail="Resume ID cannot be empty")

        result = operations.delete_resume(resume_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/", response_model=List[Dict[str, Any]])
async def list_resumes(
    skip: int = 0,
    limit: int = 10,
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """List all resumes with pagination"""
    try:
        if skip < 0:
            raise HTTPException(
                status_code=400, detail="Skip parameter cannot be negative"
            )

        if limit <= 0 or limit > 100:
            raise HTTPException(
                status_code=400, detail="Limit must be between 1 and 100"
            )

        result = operations.list_resumes(skip, limit)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/update-embeddings", response_model=Dict[str, str])
async def update_all_embeddings(
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Update vector embeddings for all resumes"""
    try:
        result = operations.update_all_vector_embeddings()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
