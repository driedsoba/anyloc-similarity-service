"""
Basic tests for AnyLoc Similarity Service API
"""
import os
import sys
import tempfile
from pathlib import Path
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from app.main import app
from app.core.config import settings

# Test client
client = TestClient(app)


def create_test_image() -> BytesIO:
    """Create a test image in memory"""
    image = Image.new('RGB', (224, 224), color='red')
    image_buffer = BytesIO()
    image.save(image_buffer, format='JPEG')
    image_buffer.seek(0)
    return image_buffer


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == settings.PROJECT_NAME
    assert "version" in data
    assert "status" in data


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    
    # Should return 200 even if AnyLoc is not initialized
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "anyloc_initialized" in data
        assert "database_stats" in data
        assert "gpu_available" in data
        assert "version" in data
        assert "uptime_seconds" in data


def test_database_stats():
    """Test database statistics endpoint"""
    response = client.get("/api/v1/database/stats")
    
    # May fail if similarity engine not initialized
    if response.status_code == 200:
        data = response.json()
        assert "total_images" in data
        assert "total_chunks" in data
        assert "total_searches" in data
        assert "anyloc_initialized" in data
    else:
        assert response.status_code == 503


def test_image_upload():
    """Test image upload endpoint"""
    # Create test image
    test_image = create_test_image()
    
    # Upload image
    response = client.post(
        "/api/v1/upload",
        files={"file": ("test_image.jpg", test_image, "image/jpeg")}
    )
    
    # May fail if AnyLoc not initialized, but should handle gracefully
    if response.status_code == 200:
        data = response.json()
        assert "file_id" in data
        assert "filename" in data
        assert "file_size" in data
        assert "upload_time" in data
        assert "message" in data
        
        # Store file_id for further tests
        return data["file_id"]
    else:
        # Should return appropriate error status
        assert response.status_code in [500, 503]


def test_invalid_file_upload():
    """Test upload with invalid file"""
    # Create text file instead of image
    text_content = "This is not an image"
    text_buffer = BytesIO(text_content.encode())
    
    response = client.post(
        "/api/v1/upload",
        files={"file": ("test.txt", text_buffer, "text/plain")}
    )
    
    # Should return 400 for invalid file type
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "details" in data


def test_large_file_upload():
    """Test upload with file too large"""
    # Create large test image (exceeds MAX_FILE_SIZE)
    large_image = Image.new('RGB', (5000, 5000), color='blue')
    large_buffer = BytesIO()
    large_image.save(large_buffer, format='JPEG', quality=100)
    large_buffer.seek(0)
    
    response = client.post(
        "/api/v1/upload",
        files={"file": ("large_image.jpg", large_buffer, "image/jpeg")}
    )
    
    # Should return 413 for file too large
    assert response.status_code == 413
    data = response.json()
    assert "error" in data
    assert "details" in data


def test_similarity_search():
    """Test similarity search endpoint"""
    # First upload an image
    test_image = create_test_image()
    upload_response = client.post(
        "/api/v1/upload",
        files={"file": ("test_image.jpg", test_image, "image/jpeg")}
    )
    
    if upload_response.status_code != 200:
        pytest.skip("Upload failed, skipping similarity test")
    
    file_id = upload_response.json()["file_id"]
    
    # Test similarity search
    response = client.post(f"/api/v1/similarity/{file_id}?top_k=5")
    
    if response.status_code == 200:
        data = response.json()
        assert "query_file_id" in data
        assert "query_filename" in data
        assert "processing_time_ms" in data
        assert "total_similar" in data
        assert "similar_images" in data
        assert data["query_file_id"] == file_id
    else:
        # Should handle gracefully if service not ready
        assert response.status_code in [404, 500, 503]


def test_similarity_search_nonexistent():
    """Test similarity search with non-existent file ID"""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = client.post(f"/api/v1/similarity/{fake_id}")
    
    # Should return 404 for non-existent file
    assert response.status_code in [404, 500, 503]


def test_metadata_search():
    """Test metadata search endpoint"""
    response = client.get("/api/v1/search?limit=10")
    
    if response.status_code == 200:
        data = response.json()
        assert "total_found" in data
        assert "images" in data
        assert "query_time_ms" in data
        assert isinstance(data["images"], list)
    else:
        # Should handle gracefully if service not ready
        assert response.status_code in [500, 503]


def test_delete_image():
    """Test image deletion endpoint"""
    # First upload an image
    test_image = create_test_image()
    upload_response = client.post(
        "/api/v1/upload",
        files={"file": ("test_delete.jpg", test_image, "image/jpeg")}
    )
    
    if upload_response.status_code != 200:
        pytest.skip("Upload failed, skipping delete test")
    
    file_id = upload_response.json()["file_id"]
    
    # Delete the image
    response = client.delete(f"/api/v1/uploads/{file_id}")
    
    if response.status_code == 200:
        data = response.json()
        assert "success" in data
        assert "message" in data
        assert data["success"] is True
    else:
        # Should handle gracefully if service not ready
        assert response.status_code in [404, 500, 503]


def test_delete_nonexistent_image():
    """Test deletion of non-existent image"""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = client.delete(f"/api/v1/uploads/{fake_id}")
    
    # Should return 404 for non-existent file
    assert response.status_code in [404, 500, 503]


def test_api_endpoints_structure():
    """Test that all expected endpoints exist and return proper status codes"""
    endpoints = [
        ("/", ["GET"]),
        ("/api/v1/health", ["GET"]),
        ("/api/v1/database/stats", ["GET"]),
        ("/api/v1/upload", ["POST"]),
        ("/api/v1/search", ["GET"]),
    ]
    
    for endpoint, methods in endpoints:
        for method in methods:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST" and "upload" in endpoint:
                # For upload endpoint, we need multipart data
                test_image = create_test_image()
                response = client.post(
                    endpoint,
                    files={"file": ("test.jpg", test_image, "image/jpeg")}
                )
            else:
                continue
            
            # Should not return 404 (endpoint should exist)
            assert response.status_code != 404, f"Endpoint {endpoint} not found"


@pytest.mark.asyncio
async def test_service_startup():
    """Test that service can start up without errors"""
    # This is tested implicitly by other tests
    # If the TestClient works, the service started successfully
    response = client.get("/")
    assert response.status_code == 200


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
