"""
Utility functions for AnyLoc Similarity Service
"""
import os
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_image_info(image_path: Path) -> Dict[str, Any]:
    """Extract basic information from an image file"""
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
                "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            }
    except Exception as e:
        logger.error(f"Failed to get image info for {image_path}: {e}")
        return {}


def validate_image_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validate if file is a valid image
    Returns (is_valid, error_message)
    """
    try:
        if not file_path.exists():
            return False, "File does not exist"
        
        if file_path.stat().st_size == 0:
            return False, "File is empty"
        
        # Try to open with PIL
        with Image.open(file_path) as img:
            # Verify image can be loaded
            img.verify()
            
        # Re-open to get image info (verify() invalidates the image)
        with Image.open(file_path) as img:
            if img.width == 0 or img.height == 0:
                return False, "Image has zero dimensions"
            
            if img.width > 10000 or img.height > 10000:
                return False, "Image dimensions too large"
        
        return True, "Valid image"
        
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def ensure_directory_exists(directory: Path) -> None:
    """Ensure directory exists, create if it doesn't"""
    directory.mkdir(parents=True, exist_ok=True)


def safe_delete_file(file_path: Path) -> bool:
    """Safely delete a file, return True if successful"""
    try:
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        return False


def get_available_disk_space(path: Path) -> int:
    """Get available disk space in bytes for given path"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        return free
    except Exception as e:
        logger.error(f"Failed to get disk space for {path}: {e}")
        return 0


def normalize_filename(filename: str) -> str:
    """Normalize filename by removing invalid characters"""
    import re
    # Remove invalid characters for filesystem
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    return filename


def generate_thumbnail(image_path: Path, thumbnail_path: Path, size: Tuple[int, int] = (224, 224)) -> bool:
    """Generate thumbnail for an image"""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(thumbnail_path, format='JPEG', quality=85)
        return True
    except Exception as e:
        logger.error(f"Failed to generate thumbnail for {image_path}: {e}")
        return False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    try:
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    except Exception as e:
        logger.error(f"Failed to calculate cosine similarity: {e}")
        return 0.0


def batch_cosine_similarity(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between query and multiple database vectors"""
    try:
        # Normalize vectors
        query_norm = query / np.linalg.norm(query)
        db_norms = np.linalg.norm(database, axis=1, keepdims=True)
        db_normalized = database / (db_norms + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Calculate similarities
        similarities = np.dot(db_normalized, query_norm)
        return similarities
    except Exception as e:
        logger.error(f"Failed to calculate batch cosine similarity: {e}")
        return np.array([])


def setup_logging(log_dir: Path, log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "anyloc_service.log")
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def log_performance(func_name: str, start_time: float, end_time: float) -> None:
    """Log performance metrics for a function"""
    duration = end_time - start_time
    logger.info(f"⏱️  {func_name} completed in {format_duration(duration)}")


class PerformanceTimer:
    """Context manager for performance timing"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        logger.info(f"⏱️  {self.operation_name} completed in {format_duration(duration)}")


def memory_usage_mb() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        logger.warning("psutil not available, cannot get memory usage")
        return 0.0
    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        return 0.0
