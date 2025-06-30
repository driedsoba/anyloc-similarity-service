"""
SQLite-based similarity engine for AnyLoc features
"""
import os
import sqlite3
import json
import asyncio
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from app.core.config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


class SQLiteSimilarityEngine:
    """
    SQLite-based similarity engine for AnyLoc features
    Handles feature extraction, storage, and similarity search
    """
    
    def __init__(self):
        self.db_path = settings.DATABASE_PATH
        self.features_dir = settings.FEATURES_DIR
        self.device = settings.get_device()
        self.executor = ThreadPoolExecutor(max_workers=settings.DB_POOL_SIZE)
        self.db_lock = threading.Lock()
        
        # AnyLoc components (initialized in initialize())
        self.dino_extractor = None
        self.vlad = None
        self.image_transform = None
        self.initialized = False
        
        # Initialize database
        self._init_database()
        
        logger.info(f"SQLiteSimilarityEngine initialized with device: {self.device}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database with tables and indexes"""
        try:
            with sqlite3.connect(str(self.db_path), timeout=settings.DB_TIMEOUT) as conn:
                if settings.ENABLE_WAL_MODE:
                    conn.execute("PRAGMA journal_mode=WAL")
                
                # Create images table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS images (
                        image_id TEXT PRIMARY KEY,
                        image_path TEXT NOT NULL,
                        original_filename TEXT,
                        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_size INTEGER,
                        width INTEGER,
                        height INTEGER,
                        feature_chunk_id INTEGER,
                        feature_index_in_chunk INTEGER,
                        similarity_searches INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP,
                        metadata_json TEXT,
                        tags TEXT
                    )
                """)
                
                # Create feature chunks table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feature_chunks (
                        chunk_id INTEGER PRIMARY KEY,
                        chunk_file TEXT NOT NULL,
                        image_count INTEGER DEFAULT 0,
                        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create search history table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_history (
                        search_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_image_id TEXT,
                        search_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        top_k INTEGER,
                        processing_time_ms REAL,
                        results_count INTEGER
                    )
                """)
                
                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_images_upload_time ON images(upload_time)",
                    "CREATE INDEX IF NOT EXISTS idx_images_filename ON images(original_filename)",
                    "CREATE INDEX IF NOT EXISTS idx_images_chunk ON images(feature_chunk_id)",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_modified ON feature_chunks(last_modified)",
                    "CREATE INDEX IF NOT EXISTS idx_search_time ON search_history(search_time)"
                ]
                
                for index_sql in indexes:
                    conn.execute(index_sql)
                
                conn.commit()
                logger.info("Database initialized successfully")
        
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def initialize(self) -> bool:
        """Initialize AnyLoc components (DINOv2 + VLAD)"""
        try:
            # Import AnyLoc components
            import sys
            anyloc_path = str(settings.ANYLOC_PATH)
            if anyloc_path not in sys.path:
                sys.path.append(anyloc_path)
            
            from anyloc.demo.utilities import DinoV2ExtractFeatures, VLAD
            
            # Initialize DINOv2 extractor
            self.dino_extractor = DinoV2ExtractFeatures(
                model_id=settings.DINO_MODEL,
                desc_layer=settings.DESC_LAYER,
                desc_facet=settings.DESC_FACET,
                device=self.device
            )
            
            # Load VLAD vocabulary
            if not settings.vlad_centers_path.exists():
                logger.warning(f"VLAD centers not found at {settings.vlad_centers_path}")
                return False
            
            vlad_centers = torch.load(settings.vlad_centers_path, map_location=self.device)
            self.vlad = VLAD(
                num_clusters=settings.VLAD_CLUSTERS,
                desc_dim=settings.VLAD_DESC_DIM,
                cache_dir=str(settings.MODEL_PATH / "vocabulary"),
                device=self.device
            )
            self.vlad.c_centers = vlad_centers.to(self.device)
            
            # Setup image transforms
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self.initialized = True
            logger.info("AnyLoc components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"AnyLoc initialization failed: {e}")
            self.initialized = False
            return False
    
    def extract_features(self, image_path: Path) -> np.ndarray:
        """
        Extract AnyLoc features from an image
        Returns 98,304D feature vector (1536 * 64)
        """
        if not self.initialized:
            raise RuntimeError("AnyLoc not initialized. Call initialize() first.")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # Extract DINOv2 features
            with torch.no_grad():
                dino_features = self.dino_extractor(image_tensor)  # [1, num_patches, 1536]
                
                # VLAD aggregation
                vlad_features = self.vlad(dino_features)  # [1, 98304]
                
                # Normalize and convert to numpy
                vlad_features = F.normalize(vlad_features, p=2, dim=1)
                features = vlad_features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {e}")
            raise
    
    async def add_to_database(
        self,
        image_path: Path,
        original_filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add image to database with feature extraction"""
        image_id = str(uuid.uuid4())
        
        try:
            # Extract features
            features = self.extract_features(image_path)
            
            # Get image metadata
            with Image.open(image_path) as img:
                width, height = img.size
            file_size = image_path.stat().st_size
            
            # Find or create feature chunk
            chunk_id, index_in_chunk = await self._add_to_chunk(features)
            
            # Insert into database
            await self._insert_image_record(
                image_id=image_id,
                image_path=str(image_path),
                original_filename=original_filename,
                file_size=file_size,
                width=width,
                height=height,
                chunk_id=chunk_id,
                index_in_chunk=index_in_chunk,
                metadata=metadata
            )
            
            logger.info(f"Added image {image_id} to database")
            return image_id
            
        except Exception as e:
            logger.error(f"Failed to add image to database: {e}")
            raise
    
    async def _add_to_chunk(self, features: np.ndarray) -> Tuple[int, int]:
        """Add features to appropriate chunk file"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._add_to_chunk_sync, features)
    
    def _add_to_chunk_sync(self, features: np.ndarray) -> Tuple[int, int]:
        """Synchronous version of _add_to_chunk"""
        with self.db_lock:
            try:
                with sqlite3.connect(str(self.db_path), timeout=settings.DB_TIMEOUT) as conn:
                    cursor = conn.cursor()
                    
                    # Find chunk with space
                    cursor.execute(
                        "SELECT chunk_id, image_count FROM feature_chunks WHERE image_count < ? ORDER BY chunk_id DESC LIMIT 1",
                        (settings.MAX_IMAGES_PER_CHUNK,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        chunk_id, current_count = row
                        chunk_file = self.features_dir / f"chunk_{chunk_id}.npy"
                        
                        # Load existing chunk
                        if chunk_file.exists():
                            chunk_features = np.load(chunk_file)
                        else:
                            chunk_features = np.zeros((settings.MAX_IMAGES_PER_CHUNK, settings.FEATURE_DIM), dtype=np.float32)
                        
                        # Add new features
                        chunk_features[current_count] = features.astype(np.float32)
                        np.save(chunk_file, chunk_features)
                        
                        # Update chunk record
                        cursor.execute(
                            "UPDATE feature_chunks SET image_count = ?, last_modified = CURRENT_TIMESTAMP WHERE chunk_id = ?",
                            (current_count + 1, chunk_id)
                        )
                        
                        return chunk_id, current_count
                    
                    else:
                        # Create new chunk
                        cursor.execute(
                            "INSERT INTO feature_chunks (chunk_file, image_count) VALUES (?, ?)",
                            (f"chunk_{cursor.lastrowid or 0}.npy", 1)
                        )
                        chunk_id = cursor.lastrowid
                        
                        # Create chunk file
                        chunk_file = self.features_dir / f"chunk_{chunk_id}.npy"
                        chunk_features = np.zeros((settings.MAX_IMAGES_PER_CHUNK, settings.FEATURE_DIM), dtype=np.float32)
                        chunk_features[0] = features.astype(np.float32)
                        np.save(chunk_file, chunk_features)
                        
                        # Update chunk record with correct filename
                        cursor.execute(
                            "UPDATE feature_chunks SET chunk_file = ? WHERE chunk_id = ?",
                            (f"chunk_{chunk_id}.npy", chunk_id)
                        )
                        
                        conn.commit()
                        return chunk_id, 0
                        
            except Exception as e:
                logger.error(f"Error adding to chunk: {e}")
                raise
    
    async def _insert_image_record(
        self,
        image_id: str,
        image_path: str,
        original_filename: str,
        file_size: int,
        width: int,
        height: int,
        chunk_id: int,
        index_in_chunk: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert image record into database"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._insert_image_record_sync,
            image_id, image_path, original_filename, file_size,
            width, height, chunk_id, index_in_chunk, metadata
        )
    
    def _insert_image_record_sync(
        self,
        image_id: str,
        image_path: str,
        original_filename: str,
        file_size: int,
        width: int,
        height: int,
        chunk_id: int,
        index_in_chunk: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Synchronous version of _insert_image_record"""
        try:
            with sqlite3.connect(str(self.db_path), timeout=settings.DB_TIMEOUT) as conn:
                conn.execute(
                    """
                    INSERT INTO images (
                        image_id, image_path, original_filename, file_size, width, height,
                        feature_chunk_id, feature_index_in_chunk, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        image_id, image_path, original_filename, file_size, width, height,
                        chunk_id, index_in_chunk, json.dumps(metadata) if metadata else None
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error inserting image record: {e}")
            raise
    
    async def find_similar(
        self,
        query_image_id: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar images using cosine similarity"""
        start_time = datetime.now()
        
        try:
            # Get query image features
            query_features = await self._get_image_features(query_image_id)
            if query_features is None:
                raise ValueError(f"Image {query_image_id} not found")
            
            # Load all features and compute similarities
            similarities = await self._compute_similarities(query_features, query_image_id)
            
            # Get top-k results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Get image metadata for results
            results = await self._get_results_metadata(top_indices, similarities)
            
            # Record search in history
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._record_search(query_image_id, top_k, processing_time, len(results))
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    async def _get_image_features(self, image_id: str) -> Optional[np.ndarray]:
        """Get features for a specific image"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._get_image_features_sync, image_id)
    
    def _get_image_features_sync(self, image_id: str) -> Optional[np.ndarray]:
        """Synchronous version of _get_image_features"""
        try:
            with sqlite3.connect(str(self.db_path), timeout=settings.DB_TIMEOUT) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT feature_chunk_id, feature_index_in_chunk FROM images WHERE image_id = ?",
                    (image_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                chunk_id, index_in_chunk = row
                chunk_file = self.features_dir / f"chunk_{chunk_id}.npy"
                
                if not chunk_file.exists():
                    logger.error(f"Chunk file not found: {chunk_file}")
                    return None
                
                chunk_features = np.load(chunk_file)
                return chunk_features[index_in_chunk]
                
        except Exception as e:
            logger.error(f"Error getting image features: {e}")
            return None
    
    async def _compute_similarities(self, query_features: np.ndarray, query_image_id: str) -> np.ndarray:
        """Compute cosine similarities against all database features"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._compute_similarities_sync, query_features, query_image_id
        )
    
    def _compute_similarities_sync(self, query_features: np.ndarray, query_image_id: str) -> np.ndarray:
        """Synchronous version of _compute_similarities"""
        try:
            similarities = []
            
            with sqlite3.connect(str(self.db_path), timeout=settings.DB_TIMEOUT) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_id, image_count FROM feature_chunks ORDER BY chunk_id")
                chunks = cursor.fetchall()
                
                # Normalize query features
                query_norm = query_features / np.linalg.norm(query_features)
                
                for chunk_id, image_count in chunks:
                    chunk_file = self.features_dir / f"chunk_{chunk_id}.npy"
                    if not chunk_file.exists():
                        continue
                    
                    chunk_features = np.load(chunk_file)[:image_count]
                    
                    # Normalize chunk features
                    chunk_norms = np.linalg.norm(chunk_features, axis=1, keepdims=True)
                    chunk_normalized = chunk_features / (chunk_norms + 1e-8)
                    
                    # Compute cosine similarities
                    chunk_similarities = np.dot(chunk_normalized, query_norm)
                    similarities.extend(chunk_similarities)
                
                return np.array(similarities)
                
        except Exception as e:
            logger.error(f"Error computing similarities: {e}")
            return np.array([])
    
    async def _get_results_metadata(
        self, top_indices: np.ndarray, similarities: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Get metadata for top similar images"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._get_results_metadata_sync, top_indices, similarities
        )
    
    def _get_results_metadata_sync(
        self, top_indices: np.ndarray, similarities: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Synchronous version of _get_results_metadata"""
        try:
            results = []
            
            with sqlite3.connect(str(self.db_path), timeout=settings.DB_TIMEOUT) as conn:
                cursor = conn.cursor()
                
                # Get all images with their chunk info
                cursor.execute("""
                    SELECT image_id, image_path, original_filename, upload_time, 
                           file_size, width, height, feature_chunk_id, feature_index_in_chunk,
                           metadata_json, tags
                    FROM images ORDER BY feature_chunk_id, feature_index_in_chunk
                """)
                
                all_images = cursor.fetchall()
                
                for idx in top_indices:
                    if idx < len(all_images):
                        image_data = all_images[idx]
                        similarity_score = float(similarities[idx])
                        
                        # Parse metadata
                        metadata = {}
                        if image_data[9]:  # metadata_json
                            try:
                                metadata = json.loads(image_data[9])
                            except:
                                pass
                        
                        result = {
                            "image_id": image_data[0],
                            "image_path": image_data[1],
                            "original_filename": image_data[2],
                            "upload_time": image_data[3],
                            "file_size": image_data[4],
                            "width": image_data[5],
                            "height": image_data[6],
                            "similarity_score": similarity_score,
                            "metadata": metadata,
                            "tags": image_data[10]
                        }
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting results metadata: {e}")
            return []
    
    async def _record_search(
        self, query_image_id: str, top_k: int, processing_time_ms: float, results_count: int
    ) -> None:
        """Record search in history for analytics"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor, self._record_search_sync,
            query_image_id, top_k, processing_time_ms, results_count
        )
    
    def _record_search_sync(
        self, query_image_id: str, top_k: int, processing_time_ms: float, results_count: int
    ) -> None:
        """Synchronous version of _record_search"""
        try:
            with sqlite3.connect(str(self.db_path), timeout=settings.DB_TIMEOUT) as conn:
                # Record search history
                conn.execute(
                    """
                    INSERT INTO search_history (query_image_id, top_k, processing_time_ms, results_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (query_image_id, top_k, processing_time_ms, results_count)
                )
                
                # Update image access statistics
                conn.execute(
                    """
                    UPDATE images 
                    SET similarity_searches = similarity_searches + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE image_id = ?
                    """,
                    (query_image_id,)
                )
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error recording search: {e}")
    
    async def search_by_metadata(
        self,
        filename: Optional[str] = None,
        upload_after: Optional[datetime] = None,
        upload_before: Optional[datetime] = None,
        min_file_size: Optional[int] = None,
        max_file_size: Optional[int] = None,
        tags: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search images by metadata filters"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._search_by_metadata_sync,
            filename, upload_after, upload_before, min_file_size, max_file_size, tags, limit
        )
    
    def _search_by_metadata_sync(
        self,
        filename: Optional[str] = None,
        upload_after: Optional[datetime] = None,
        upload_before: Optional[datetime] = None,
        min_file_size: Optional[int] = None,
        max_file_size: Optional[int] = None,
        tags: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Synchronous version of search_by_metadata"""
        try:
            with sqlite3.connect(str(self.db_path), timeout=settings.DB_TIMEOUT) as conn:
                cursor = conn.cursor()
                
                # Build dynamic query
                conditions = []
                params = []
                
                if filename:
                    conditions.append("original_filename LIKE ?")
                    params.append(f"%{filename}%")
                
                if upload_after:
                    conditions.append("upload_time >= ?")
                    params.append(upload_after.isoformat())
                
                if upload_before:
                    conditions.append("upload_time <= ?")
                    params.append(upload_before.isoformat())
                
                if min_file_size:
                    conditions.append("file_size >= ?")
                    params.append(min_file_size)
                
                if max_file_size:
                    conditions.append("file_size <= ?")
                    params.append(max_file_size)
                
                if tags:
                    conditions.append("tags LIKE ?")
                    params.append(f"%{tags}%")
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                params.append(limit)
                
                query = f"""
                    SELECT image_id, image_path, original_filename, upload_time,
                           file_size, width, height, metadata_json, tags
                    FROM images
                    WHERE {where_clause}
                    ORDER BY upload_time DESC
                    LIMIT ?
                """
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    metadata = {}
                    if row[7]:  # metadata_json
                        try:
                            metadata = json.loads(row[7])
                        except:
                            pass
                    
                    result = {
                        "image_id": row[0],
                        "image_path": row[1],
                        "original_filename": row[2],
                        "upload_time": row[3],
                        "file_size": row[4],
                        "width": row[5],
                        "height": row[6],
                        "metadata": metadata,
                        "tags": row[8]
                    }
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._get_database_stats_sync)
    
    def _get_database_stats_sync(self) -> Dict[str, Any]:
        """Synchronous version of get_database_stats"""
        try:
            with sqlite3.connect(str(self.db_path), timeout=settings.DB_TIMEOUT) as conn:
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM images")
                total_images = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM feature_chunks")
                total_chunks = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM search_history")
                total_searches = cursor.fetchone()[0]
                
                # Storage stats
                cursor.execute("SELECT SUM(file_size) FROM images")
                total_storage = cursor.fetchone()[0] or 0
                
                # Recent activity
                cursor.execute("""
                    SELECT COUNT(*) FROM images 
                    WHERE upload_time >= datetime('now', '-24 hours')
                """)
                recent_uploads = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) FROM search_history 
                    WHERE search_time >= datetime('now', '-24 hours')
                """)
                recent_searches = cursor.fetchone()[0]
                
                # Performance stats
                cursor.execute("""
                    SELECT AVG(processing_time_ms), MAX(processing_time_ms)
                    FROM search_history 
                    WHERE search_time >= datetime('now', '-7 days')
                """)
                perf_stats = cursor.fetchone()
                avg_processing_time = perf_stats[0] or 0
                max_processing_time = perf_stats[1] or 0
                
                return {
                    "total_images": total_images,
                    "total_chunks": total_chunks,
                    "total_searches": total_searches,
                    "total_storage_bytes": total_storage,
                    "recent_uploads_24h": recent_uploads,
                    "recent_searches_24h": recent_searches,
                    "avg_processing_time_ms": round(avg_processing_time, 2),
                    "max_processing_time_ms": round(max_processing_time, 2),
                    "anyloc_initialized": self.initialized,
                    "device": str(self.device),
                    "feature_dimension": settings.FEATURE_DIM,
                    "max_images_per_chunk": settings.MAX_IMAGES_PER_CHUNK
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                "total_images": 0,
                "total_chunks": 0,
                "total_searches": 0,
                "error": str(e)
            }
    
    async def delete_image(self, image_id: str) -> bool:
        """Delete image and its features from database"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._delete_image_sync, image_id)
    
    def _delete_image_sync(self, image_id: str) -> bool:
        """Synchronous version of delete_image"""
        try:
            with sqlite3.connect(str(self.db_path), timeout=settings.DB_TIMEOUT) as conn:
                cursor = conn.cursor()
                
                # Get image info
                cursor.execute(
                    "SELECT image_path, feature_chunk_id, feature_index_in_chunk FROM images WHERE image_id = ?",
                    (image_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return False
                
                image_path, chunk_id, index_in_chunk = row
                
                # Delete image file
                try:
                    Path(image_path).unlink(missing_ok=True)
                except:
                    pass
                
                # Delete from database
                cursor.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
                cursor.execute("DELETE FROM search_history WHERE query_image_id = ?", (image_id,))
                
                # Update chunk count
                cursor.execute(
                    "UPDATE feature_chunks SET image_count = image_count - 1 WHERE chunk_id = ?",
                    (chunk_id,)
                )
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error deleting image: {e}")
            return False
    
    async def backup_database(self, backup_path: Path) -> bool:
        """Create database backup"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._backup_database_sync, backup_path)
    
    def _backup_database_sync(self, backup_path: Path) -> bool:
        """Synchronous version of backup_database"""
        try:
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup SQLite database
            with sqlite3.connect(str(self.db_path)) as source:
                with sqlite3.connect(str(backup_path / "similarity.db")) as backup:
                    source.backup(backup)
            
            # Copy feature chunks
            features_backup = backup_path / "features"
            features_backup.mkdir(exist_ok=True)
            
            for chunk_file in self.features_dir.glob("chunk_*.npy"):
                shutil.copy2(chunk_file, features_backup / chunk_file.name)
            
            logger.info(f"Database backup created at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
