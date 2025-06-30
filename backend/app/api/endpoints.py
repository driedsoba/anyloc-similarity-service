"""
FastAPI endpoints for AnyLoc Similarity Service
"""
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from app.api.models import (
    UploadResponse, SimilarityResponse, HealthResponse, ErrorResponse,
    DatabaseStats, MetadataSearchResponse, DeleteResponse, SimilarImage
)
from app.core.config import settings
from app.core.similarity import SQLiteSimilarityEngine

# Configure logging
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter()

# Global similarity engine instance (initialized in main.py)
similarity_engine: Optional[SQLiteSimilarityEngine] = None

# Service start time for uptime calculation
service_start_time = time.time()


def get_similarity_engine() -> SQLiteSimilarityEngine:
    """Dependency to get similarity engine instance"""
    if similarity_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Similarity engine not initialized"
        )
    return similarity_engine


def validate_file_type(filename: str) -> bool:
    """Validate if file type is supported"""
    file_ext = Path(filename).suffix.lower()
    return file_ext in settings.ALLOWED_EXTENSIONS


def validate_file_size(file_size: int) -> bool:
    """Validate if file size is within limits"""
    return file_size <= settings.MAX_FILE_SIZE


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns service status, AnyLoc initialization status, and database statistics
    """
    try:
        # Get database stats
        engine = get_similarity_engine()
        db_stats = await engine.get_database_stats()
        
        # Calculate uptime
        uptime = time.time() - service_start_time
        
        # Check GPU availability
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except:
            gpu_available = False
        
        # Create database stats model
        database_stats = DatabaseStats(**db_stats)
        
        return HealthResponse(
            status="healthy",
            anyloc_initialized=engine.initialized,
            database_stats=database_stats,
            gpu_available=gpu_available,
            version=settings.VERSION,
            uptime_seconds=round(uptime, 2)
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    engine: SQLiteSimilarityEngine = Depends(get_similarity_engine)
):
    """
    Upload an image and extract AnyLoc features
    Returns file ID for subsequent similarity searches
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename provided"
            )
        
        if not validate_file_type(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Supported: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Read file content
        content = await file.read()
        
        if not validate_file_size(len(content)):
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Generate unique file ID and save file
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix.lower()
        file_path = settings.UPLOAD_DIR / f"{file_id}{file_ext}"
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Add to database with feature extraction
        upload_time = datetime.now()
        actual_file_id = await engine.add_to_database(
            image_path=file_path,
            original_filename=file.filename,
            metadata={
                "upload_ip": "unknown",  # Could be extracted from request
                "file_extension": file_ext,
                "content_type": file.content_type
            }
        )
        
        logger.info(f"Successfully uploaded and processed image: {file.filename}")
        
        return UploadResponse(
            file_id=actual_file_id,
            filename=file.filename,
            file_size=len(content),
            upload_time=upload_time,
            message="Image uploaded and processed successfully"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        
        # Clean up file if it was created
        try:
            if 'file_path' in locals() and file_path.exists():
                file_path.unlink()
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Upload processing failed: {str(e)}"
        )


@router.post("/similarity/{file_id}", response_model=SimilarityResponse)
async def find_similar_images(
    file_id: str,
    top_k: int = Query(10, ge=1, le=100, description="Number of similar images to return"),
    engine: SQLiteSimilarityEngine = Depends(get_similarity_engine)
):
    """
    Find images similar to the uploaded image
    Uses cosine similarity on AnyLoc features
    """
    try:
        start_time = time.time()
        
        # Check if AnyLoc is initialized
        if not engine.initialized:
            raise HTTPException(
                status_code=503,
                detail="AnyLoc models not initialized. Please wait or check logs."
            )
        
        # Find similar images
        similar_results = await engine.find_similar(file_id, top_k)
        
        if not similar_results:
            raise HTTPException(
                status_code=404,
                detail=f"Image with ID {file_id} not found"
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert results to response models
        similar_images = []
        query_filename = "unknown"
        
        for result in similar_results:
            # Skip the query image itself (similarity = 1.0)
            if result["image_id"] == file_id:
                query_filename = result["original_filename"]
                continue
            
            similar_image = SimilarImage(
                image_id=result["image_id"],
                image_path=result["image_path"],
                original_filename=result["original_filename"],
                similarity_score=result["similarity_score"],
                upload_time=datetime.fromisoformat(result["upload_time"].replace('Z', '+00:00')) if isinstance(result["upload_time"], str) else result["upload_time"],
                file_size=result["file_size"],
                width=result["width"],
                height=result["height"],
                metadata=result.get("metadata"),
                tags=result.get("tags")
            )
            similar_images.append(similar_image)
        
        logger.info(f"Found {len(similar_images)} similar images for {file_id}")
        
        return SimilarityResponse(
            query_file_id=file_id,
            query_filename=query_filename,
            processing_time_ms=round(processing_time, 2),
            total_similar=len(similar_images),
            similar_images=similar_images
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Similarity search failed: {str(e)}"
        )


@router.get("/database/stats", response_model=DatabaseStats)
async def get_database_statistics(
    engine: SQLiteSimilarityEngine = Depends(get_similarity_engine)
):
    """
    Get comprehensive database statistics
    Includes counts, storage usage, and performance metrics
    """
    try:
        stats = await engine.get_database_stats()
        return DatabaseStats(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve database statistics: {str(e)}"
        )


@router.delete("/uploads/{file_id}", response_model=DeleteResponse)
async def delete_image(
    file_id: str,
    engine: SQLiteSimilarityEngine = Depends(get_similarity_engine)
):
    """
    Delete an uploaded image and its features from the database
    This action cannot be undone
    """
    try:
        success = await engine.delete_image(file_id)
        
        if success:
            logger.info(f"Successfully deleted image: {file_id}")
            return DeleteResponse(
                success=True,
                message="Image and associated data deleted successfully",
                deleted_file_id=file_id
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Image with ID {file_id} not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Delete operation failed: {str(e)}"
        )


@router.get("/search", response_model=MetadataSearchResponse)
async def search_by_metadata(
    filename: Optional[str] = Query(None, description="Filter by filename (partial match)"),
    upload_after: Optional[datetime] = Query(None, description="Filter by upload time (after this date)"),
    upload_before: Optional[datetime] = Query(None, description="Filter by upload time (before this date)"),
    min_file_size: Optional[int] = Query(None, ge=0, description="Minimum file size in bytes"),
    max_file_size: Optional[int] = Query(None, ge=0, description="Maximum file size in bytes"),
    tags: Optional[str] = Query(None, description="Filter by tags (partial match)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    engine: SQLiteSimilarityEngine = Depends(get_similarity_engine)
):
    """
    Search images by metadata filters
    Supports filtering by filename, upload time, file size, and tags
    """
    try:
        start_time = time.time()
        
        # Perform metadata search
        results = await engine.search_by_metadata(
            filename=filename,
            upload_after=upload_after,
            upload_before=upload_before,
            min_file_size=min_file_size,
            max_file_size=max_file_size,
            tags=tags,
            limit=limit
        )
        
        query_time = (time.time() - start_time) * 1000
        
        # Convert results to response models
        images = []
        for result in results:
            similar_image = SimilarImage(
                image_id=result["image_id"],
                image_path=result["image_path"],
                original_filename=result["original_filename"],
                similarity_score=1.0,  # Not applicable for metadata search
                upload_time=datetime.fromisoformat(result["upload_time"].replace('Z', '+00:00')) if isinstance(result["upload_time"], str) else result["upload_time"],
                file_size=result["file_size"],
                width=result["width"],
                height=result["height"],
                metadata=result.get("metadata"),
                tags=result.get("tags")
            )
            images.append(similar_image)
        
        logger.info(f"Metadata search returned {len(images)} results")
        
        return MetadataSearchResponse(
            total_found=len(images),
            images=images,
            query_time_ms=round(query_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Metadata search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Metadata search failed: {str(e)}"
        )


@router.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health"
    }
