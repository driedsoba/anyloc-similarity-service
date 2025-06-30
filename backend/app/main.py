"""
FastAPI main application for AnyLoc Similarity Service
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.similarity import SQLiteSimilarityEngine
from app.api import endpoints
from app.api.models import ErrorResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.LOG_DIR / "anyloc_service.log")
    ]
)
logger = logging.getLogger(__name__)

# Global similarity engine instance
similarity_engine: SQLiteSimilarityEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app
    Handles startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting AnyLoc Similarity Service...")
    logger.info(f"📍 Base directory: {settings.BASE_DIR}")
    logger.info(f"🎯 Device: {settings.DEVICE}")
    logger.info(f"🗃️  Database: {settings.DATABASE_PATH}")
    
    try:
        # Initialize similarity engine
        global similarity_engine
        similarity_engine = SQLiteSimilarityEngine()
        
        # Initialize AnyLoc models
        logger.info("🤖 Initializing AnyLoc models...")
        init_success = await similarity_engine.initialize()
        
        if init_success:
            logger.info("✅ AnyLoc models initialized successfully")
        else:
            logger.warning("⚠️  AnyLoc initialization failed - service will run with limited functionality")
        
        # Make engine available to endpoints
        endpoints.similarity_engine = similarity_engine
        
        logger.info("🌐 Service started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AnyLoc Similarity Service...")
    
    if similarity_engine and hasattr(similarity_engine, 'executor'):
        similarity_engine.executor.shutdown(wait=True)
    
    logger.info("👋 Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    A production-ready FastAPI backend service for universal visual place recognition 
    and image similarity search using AnyLoc (IEEE RA-L 2023).
    
    Features:
    - Upload images and extract AnyLoc features (DINOv2 + VLAD)
    - Find similar images using cosine similarity search
    - SQLite database with chunked feature storage
    - Search by metadata (filename, upload time, file size, tags)
    - Comprehensive analytics and statistics
    
    Built with FastAPI, PyTorch, and AnyLoc research models.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Include API routes
app.include_router(
    endpoints.router,
    prefix=settings.API_V1_STR,
    tags=["anyloc-similarity"]
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Global HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            details=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            details="An unexpected error occurred. Please check the logs.",
            timestamp=datetime.now()
        ).dict()
    )


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "description": "AnyLoc Similarity Service - Universal Visual Place Recognition",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": f"{settings.API_V1_STR}/health",
        "api": settings.API_V1_STR,
        "features": [
            "Image upload and feature extraction",
            "Similarity search with AnyLoc features",
            "Metadata-based search",
            "Database analytics",
            "RESTful API with automatic documentation"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
