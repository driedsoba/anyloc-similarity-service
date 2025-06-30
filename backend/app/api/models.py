"""
Pydantic models for API request/response schemas
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response model for image upload"""
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    upload_time: datetime = Field(..., description="Upload timestamp")
    message: str = Field(..., description="Success message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "vacation_photo.jpg",
                "file_size": 2048576,
                "upload_time": "2023-12-01T10:30:00Z",
                "message": "Image uploaded and processed successfully"
            }
        }


class SimilarImage(BaseModel):
    """Model for a similar image result"""
    image_id: str = Field(..., description="Unique identifier of the similar image")
    image_path: str = Field(..., description="Path to the similar image")
    original_filename: str = Field(..., description="Original filename of the similar image")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score (0-1)")
    upload_time: datetime = Field(..., description="When the similar image was uploaded")
    file_size: int = Field(..., description="File size in bytes")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    tags: Optional[str] = Field(None, description="Image tags")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_id": "550e8400-e29b-41d4-a716-446655440001",
                "image_path": "/uploads/550e8400-e29b-41d4-a716-446655440001.jpg",
                "original_filename": "similar_photo.jpg",
                "similarity_score": 0.87,
                "upload_time": "2023-11-30T15:45:00Z",
                "file_size": 1856432,
                "width": 1920,
                "height": 1080,
                "metadata": {"camera": "Canon EOS R5", "location": "Paris"},
                "tags": "vacation, city, architecture"
            }
        }


class SimilarityResponse(BaseModel):
    """Response model for similarity search"""
    query_file_id: str = Field(..., description="ID of the query image")
    query_filename: str = Field(..., description="Filename of the query image")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    total_similar: int = Field(..., description="Number of similar images found")
    similar_images: List[SimilarImage] = Field(..., description="List of similar images")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_file_id": "550e8400-e29b-41d4-a716-446655440000",
                "query_filename": "vacation_photo.jpg",
                "processing_time_ms": 85.6,
                "total_similar": 5,
                "similar_images": [
                    {
                        "image_id": "550e8400-e29b-41d4-a716-446655440001",
                        "image_path": "/uploads/550e8400-e29b-41d4-a716-446655440001.jpg",
                        "original_filename": "similar_photo.jpg",
                        "similarity_score": 0.87,
                        "upload_time": "2023-11-30T15:45:00Z",
                        "file_size": 1856432,
                        "width": 1920,
                        "height": 1080,
                        "metadata": {"camera": "Canon EOS R5"},
                        "tags": "vacation, city"
                    }
                ]
            }
        }


class DatabaseStats(BaseModel):
    """Database statistics model"""
    total_images: int = Field(..., description="Total number of images in database")
    total_chunks: int = Field(..., description="Total number of feature chunks")
    total_searches: int = Field(..., description="Total number of searches performed")
    total_storage_bytes: int = Field(..., description="Total storage used in bytes")
    recent_uploads_24h: int = Field(..., description="Number of uploads in last 24 hours")
    recent_searches_24h: int = Field(..., description="Number of searches in last 24 hours")
    avg_processing_time_ms: float = Field(..., description="Average processing time in milliseconds")
    max_processing_time_ms: float = Field(..., description="Maximum processing time in milliseconds")
    anyloc_initialized: bool = Field(..., description="Whether AnyLoc models are initialized")
    device: str = Field(..., description="Compute device being used")
    feature_dimension: int = Field(..., description="Feature vector dimension")
    max_images_per_chunk: int = Field(..., description="Maximum images per feature chunk")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_images": 1542,
                "total_chunks": 2,
                "total_searches": 89,
                "total_storage_bytes": 2147483648,
                "recent_uploads_24h": 23,
                "recent_searches_24h": 15,
                "avg_processing_time_ms": 92.5,
                "max_processing_time_ms": 245.8,
                "anyloc_initialized": True,
                "device": "cuda",
                "feature_dimension": 98304,
                "max_images_per_chunk": 1000
            }
        }


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    anyloc_initialized: bool = Field(..., description="Whether AnyLoc is properly initialized")
    database_stats: DatabaseStats = Field(..., description="Current database statistics")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "anyloc_initialized": True,
                "database_stats": {
                    "total_images": 1542,
                    "total_chunks": 2,
                    "total_searches": 89,
                    "anyloc_initialized": True,
                    "device": "cuda"
                },
                "gpu_available": True,
                "version": "1.0.0",
                "uptime_seconds": 3600.5
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type or category")
    details: str = Field(..., description="Detailed error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "details": "Invalid file format. Only JPG, PNG, BMP, and TIFF files are supported.",
                "timestamp": "2023-12-01T10:30:00Z"
            }
        }


class MetadataSearchRequest(BaseModel):
    """Request model for metadata-based search"""
    filename: Optional[str] = Field(None, description="Filter by filename (partial match)")
    upload_after: Optional[datetime] = Field(None, description="Filter by upload time (after)")
    upload_before: Optional[datetime] = Field(None, description="Filter by upload time (before)")
    min_file_size: Optional[int] = Field(None, ge=0, description="Minimum file size in bytes")
    max_file_size: Optional[int] = Field(None, ge=0, description="Maximum file size in bytes")
    tags: Optional[str] = Field(None, description="Filter by tags (partial match)")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "filename": "vacation",
                "upload_after": "2023-11-01T00:00:00Z",
                "upload_before": "2023-12-01T00:00:00Z",
                "min_file_size": 1000000,
                "max_file_size": 10000000,
                "tags": "travel",
                "limit": 50
            }
        }


class MetadataSearchResponse(BaseModel):
    """Response model for metadata-based search"""
    total_found: int = Field(..., description="Total number of images found")
    images: List[SimilarImage] = Field(..., description="List of matching images")
    query_time_ms: float = Field(..., description="Query processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_found": 15,
                "images": [
                    {
                        "image_id": "550e8400-e29b-41d4-a716-446655440001",
                        "image_path": "/uploads/550e8400-e29b-41d4-a716-446655440001.jpg",
                        "original_filename": "vacation_paris.jpg",
                        "similarity_score": 1.0,
                        "upload_time": "2023-11-30T15:45:00Z",
                        "file_size": 1856432,
                        "width": 1920,
                        "height": 1080,
                        "metadata": {"camera": "Canon EOS R5"},
                        "tags": "vacation, travel, paris"
                    }
                ],
                "query_time_ms": 12.5
            }
        }


class DeleteResponse(BaseModel):
    """Response model for image deletion"""
    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Success or error message")
    deleted_file_id: Optional[str] = Field(None, description="ID of the deleted file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Image and associated data deleted successfully",
                "deleted_file_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
