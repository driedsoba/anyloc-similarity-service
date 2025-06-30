"""
Configuration settings for AnyLoc Similarity Service
"""
import os
from pathlib import Path
from typing import Set, Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

try:
    import torch
except ImportError:
    torch = None


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AnyLoc Similarity Service"
    VERSION: str = "1.0.0"
    
    # File Paths
    BASE_DIR: Path = Path(__file__).parent.parent.absolute()
    MODEL_PATH: Path = BASE_DIR / "models"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    ANYLOC_PATH: Path = BASE_DIR / "anyloc"
    LOG_DIR: Path = BASE_DIR / "logs"
    DATABASE_PATH: Path = BASE_DIR / "models" / "similarity.db"
    FEATURES_DIR: Path = BASE_DIR / "models" / "features"
    
    # File Handling
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    MAX_IMAGES_PER_CHUNK: int = 1000
    
    # AnyLoc Settings
    DEVICE: str = "auto"  # auto, cuda, cpu
    DINO_MODEL: str = "dinov2_vitg14"
    DESC_LAYER: int = 31
    DESC_FACET: str = "value"
    VLAD_CLUSTERS: int = 64
    VLAD_DESC_DIM: int = 1536
    FEATURE_DIM: int = 98304  # VLAD_DESC_DIM * VLAD_CLUSTERS
    
    # Database Settings
    DB_POOL_SIZE: int = 10
    DB_TIMEOUT: int = 30
    ENABLE_WAL_MODE: bool = True
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_device()
        self._create_directories()
    
    def _setup_device(self) -> None:
        """Set up the compute device"""
        if torch is None:
            self.DEVICE = "cpu"
            print("⚠️  PyTorch not available, using CPU")
            return
            
        if self.DEVICE == "auto":
            if torch.cuda.is_available():
                self.DEVICE = "cuda"
                print(f"🎯 Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.DEVICE = "cpu"
                print("⚠️  CUDA not available, using CPU")
        else:
            print(f"🎯 Using device: {self.DEVICE}")
    
    def _create_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            self.MODEL_PATH,
            self.UPLOAD_DIR,
            self.LOG_DIR,
            self.FEATURES_DIR,
            self.MODEL_PATH / "vocabulary" / self.DINO_MODEL / f"l{self.DESC_LAYER}_{self.DESC_FACET}_c{self.VLAD_CLUSTERS}"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def vlad_centers_path(self) -> Path:
        """Path to VLAD cluster centers"""
        return self.MODEL_PATH / "vocabulary" / self.DINO_MODEL / f"l{self.DESC_LAYER}_{self.DESC_FACET}_c{self.VLAD_CLUSTERS}" / "c_centers.pt"
    
    @property
    def database_url(self) -> str:
        """SQLite database URL"""
        return f"sqlite:///{self.DATABASE_PATH}"
    
    def get_device(self):
        """Get PyTorch device"""
        if torch is None:
            return "cpu"
        return torch.device(self.DEVICE)


# Global settings instance
settings = Settings()
