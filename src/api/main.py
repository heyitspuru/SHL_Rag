"""
FastAPI application for SHL Assessment Recommendation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from datetime import datetime
import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.rag.embeddings import AssessmentEmbedder
from src.rag.vectorstore import VectorStoreManager
from src.rag.retriever import AssessmentRetriever
from src.rag.recommender import SHLRecommendationEngine
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Intelligent RAG-based system for recommending SHL assessments",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    message: str


class RecommendationRequest(BaseModel):
    query: str = Field(..., description="Job description or natural language query")
    top_k: Optional[int] = Field(10, ge=1, le=10, description="Number of recommendations (1-10)")


class Assessment(BaseModel):
    assessment_name: str
    assessment_url: str
    test_type: str
    category: str
    duration_minutes: Optional[int] = None
    description: Optional[str] = None


class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[Assessment]
    total_found: int


# Global recommendation engine (load once at startup)
recommendation_engine = None
catalog_df = None


@app.on_event("startup")
async def startup_event():
    """Initialize recommendation engine on startup"""
    global recommendation_engine, catalog_df
    
    try:
        logger.info("Initializing recommendation system...")
        
        # Load catalog
        catalog_df = pd.read_csv('data/raw/shl_assessments.csv')
        logger.info(f"✅ Loaded catalog: {len(catalog_df)} assessments")
        
        # Initialize embeddings
        embedder = AssessmentEmbedder(model_name="all-MiniLM-L6-v2")
        logger.info("✅ Embeddings loaded")
        
        # Load vector store
        vs_manager = VectorStoreManager(embedder.embeddings)
        vectorstore = vs_manager.load_vectorstore(use_chroma=True)
        logger.info("✅ Vector store loaded")
        
        # Create retriever
        retriever = AssessmentRetriever(vectorstore)
        
        # Create recommendation engine
        recommendation_engine = SHLRecommendationEngine(
            retriever=retriever,
            llm=None,  # Can add LLM here if needed
            catalog_df=catalog_df
        )
        
        logger.info("✅ Recommendation engine ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SHL Assessment Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns the current status of the API and recommendation engine.
    """
    is_ready = recommendation_engine is not None
    
    return {
        "status": "healthy" if is_ready else "initializing",
        "timestamp": datetime.now().isoformat(),
        "message": "Recommendation engine is ready" if is_ready else "Recommendation engine is starting..."
    }


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend_assessments(request: RecommendationRequest):
    """
    Get assessment recommendations for a job query.
    
    This endpoint analyzes the provided query and returns relevant SHL assessments
    with intelligent balancing across test types (Knowledge, Cognitive, Personality, Situational).
    
    **Parameters:**
    - **query**: Job description or requirements (min 10 characters)
    - **top_k**: Number of recommendations to return (1-10, default: 10)
    
    **Returns:**
    - List of recommended assessments with details
    """
    if not recommendation_engine:
        raise HTTPException(status_code=503, detail="Recommendation engine not ready")
    
    try:
        # Generate recommendations
        recommendations = recommendation_engine.recommend(
            query=request.query,
            top_k=request.top_k,
            enable_balancing=True,
            enable_reranking=False
        )
        
        # Format response
        assessments = [
            Assessment(
                assessment_name=rec['assessment_name'],
                assessment_url=rec['assessment_url'],
                test_type=rec['test_type'],
                category=rec.get('category', 'General'),
                duration_minutes=rec.get('duration_minutes'),
                description=rec.get('description', '')
            )
            for rec in recommendations
        ]
        
        return RecommendationResponse(
            query=request.query,
            recommendations=assessments,
            total_found=len(assessments)
        )
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Get statistics about the assessment catalog"""
    if catalog_df is None:
        raise HTTPException(status_code=503, detail="Catalog not loaded")
    
    type_counts = catalog_df['test_type'].value_counts().to_dict()
    category_counts = catalog_df['category'].value_counts().to_dict()
    
    return {
        "total_assessments": len(catalog_df),
        "test_types": type_counts,
        "categories": category_counts
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
