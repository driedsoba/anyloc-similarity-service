"""
Development server runner for AnyLoc Similarity Service
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

import uvicorn
from app.core.config import settings


def main():
    """Run development server"""
    print("🚀 Starting AnyLoc Similarity Service Development Server...")
    print(f"📍 Host: {settings.HOST}")
    print(f"🔌 Port: {settings.PORT}")
    print(f"🐞 Debug: {settings.DEBUG}")
    print(f"🎯 Device: {settings.DEVICE}")
    print("")
    print("📖 API Documentation:")
    print(f"   Swagger UI: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"   ReDoc: http://{settings.HOST}:{settings.PORT}/redoc")
    print(f"   Health Check: http://{settings.HOST}:{settings.PORT}/api/v1/health")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Set environment for development
    os.environ["PYTHONPATH"] = f"{current_dir.parent}:{current_dir.parent / 'anyloc'}"
    
    # Run server
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        reload_dirs=[str(current_dir.parent / "app")]
    )


if __name__ == "__main__":
    main()
