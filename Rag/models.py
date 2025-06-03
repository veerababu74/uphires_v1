from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class BestMatchResult(BaseModel):
    """Pydantic model for the LLM output"""

    id: str = Field(
        alias="_id", description="MongoDB ObjectId of the best matching candidate"
    )

    class Config:
        populate_by_name = True


class CandidateMatch(BaseModel):
    """Pydantic model for a single candidate match with score"""

    id: str = Field(alias="_id", description="MongoDB ObjectId of the candidate")
    relevance_score: float = Field(
        description="Relevance score from 0.0 to 1.0, where 1.0 is perfect match"
    )
    match_reason: str = Field(
        description="Brief explanation of why this candidate matches the query"
    )

    class Config:
        populate_by_name = True


class AllMatchesResult(BaseModel):
    """Pydantic model for all matching candidates ranked by relevance"""

    total_candidates: int = Field(description="Total number of candidates analyzed")
    matches: List[CandidateMatch] = Field(
        description="List of all matching candidates ranked by relevance score (highest first)"
    )


class SearchStatistics(BaseModel):
    """Statistics for search operations"""

    mongodb_retrieved: int = 0
    llm_context_sent: int = 0
    context_length: int = 0
    context_limit_exceeded: bool = False
    query: str = ""
    retrieved: int = 0
    analyzed: int = 0
