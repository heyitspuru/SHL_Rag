"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    message: str


class RecommendationRequest(BaseModel):
    query: str = Field(..., description="Job description or natural language query", min_length=10)
    top_k: Optional[int] = Field(10, ge=1, le=10, description="Number of recommendations (1-10)")


class Assessment(BaseModel):
    assessment_name: str
    assessment_url: str
    test_type: str
    category: Optional[str] = None
    duration: Optional[str] = None


class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[Assessment]
    count: int
    timestamp: str
