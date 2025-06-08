# resume_api/models/search.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union


class VectorSearchQuery(BaseModel):
    user_id: Optional[str] = Field(..., description="User ID who performed the search")
    query: str = Field(..., description="Search query for semantic search")
    field: Literal["full_text"] = Field(
        default="full_text",
        description="Field to search in (fixed to full_text for total resume search)",
    )
    num_results: Literal[10] = Field(
        default=10, description="Fixed number of results to return"
    )
    min_score: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Minimum similarity score threshold"
    )

    class Config:
        validate_assignment = True
