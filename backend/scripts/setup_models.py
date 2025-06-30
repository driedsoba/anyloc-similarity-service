"""
Setup script for AnyLoc models and vocabularies
Creates dummy VLAD centers for testing/development
"""
import os
import sys
from pathlib import Path
import logging

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_vlad_vocabulary():
    """
    Setup VLAD vocabulary centers
    For production, these should be downloaded from AnyLoc's official release
    For development/testing, we create dummy centers
    """
    try:
        import torch
        
        vocab_dir = settings.MODEL_PATH / "vocabulary" / settings.DINO_MODEL / f"l{settings.DESC_LAYER}_{settings.DESC_FACET}_c{settings.VLAD_CLUSTERS}"
        vocab_dir.mkdir(parents=True, exist_ok=True)
        
        centers_path = vocab_dir / "c_centers.pt"
        
        if centers_path.exists():
            logger.info(f"✅ VLAD centers already exist at {centers_path}")
            return True
        
        logger.info(f"🔧 Creating VLAD centers at {centers_path}")
        logger.warning("⚠️  Using dummy VLAD centers for development/testing")
        logger.warning("⚠️  For production, download official centers from AnyLoc repository")
        
        # Create dummy VLAD centers
        # In production, these should be loaded from the official AnyLoc release
        dummy_centers = torch.randn(settings.VLAD_CLUSTERS, settings.VLAD_DESC_DIM)
        torch.save(dummy_centers, centers_path)
        
        logger.info(f"✅ VLAD centers created successfully")
        logger.info(f"📊 Centers shape: {dummy_centers.shape}")
        logger.info(f"🎯 Clusters: {settings.VLAD_CLUSTERS}")
        logger.info(f"📐 Dimensions: {settings.VLAD_DESC_DIM}")
        
        return True
        
    except ImportError:
        logger.error("❌ PyTorch not available. Please install PyTorch first.")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to setup VLAD vocabulary: {e}")
        return False


def setup_directories():
    """Create necessary directories"""
    directories = [
        settings.MODEL_PATH,
        settings.UPLOAD_DIR,
        settings.LOG_DIR,
        settings.FEATURES_DIR,
        settings.MODEL_PATH / "vocabulary",
        settings.MODEL_PATH / "checkpoints"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")
    
    # Create .gitkeep files
    gitkeep_dirs = [
        settings.UPLOAD_DIR,
        settings.LOG_DIR,
        settings.FEATURES_DIR
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_file = directory / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()


def check_anyloc_submodule():
    """Check if AnyLoc submodule is properly initialized"""
    anyloc_path = settings.ANYLOC_PATH
    
    if not anyloc_path.exists():
        logger.error(f"❌ AnyLoc submodule not found at {anyloc_path}")
        logger.error("💡 Run: git submodule update --init --recursive")
        return False
    
    demo_utils = anyloc_path / "demo" / "utilities.py"
    if not demo_utils.exists():
        logger.error(f"❌ AnyLoc utilities not found at {demo_utils}")
        logger.error("💡 AnyLoc submodule may be incomplete")
        return False
    
    logger.info(f"✅ AnyLoc submodule found at {anyloc_path}")
    return True


def test_imports():
    """Test if required imports work"""
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} available")
        
        if torch.cuda.is_available():
            logger.info(f"🎯 CUDA available: {torch.cuda.get_device_name()}")
        else:
            logger.warning("⚠️  CUDA not available, will use CPU")
        
    except ImportError:
        logger.error("❌ PyTorch not available")
        return False
    
    try:
        import torchvision
        logger.info(f"✅ TorchVision {torchvision.__version__} available")
    except ImportError:
        logger.error("❌ TorchVision not available")
        return False
    
    try:
        from PIL import Image
        logger.info("✅ PIL/Pillow available")
    except ImportError:
        logger.error("❌ PIL/Pillow not available")
        return False
    
    try:
        import numpy as np
        logger.info(f"✅ NumPy {np.__version__} available")
    except ImportError:
        logger.error("❌ NumPy not available")
        return False
    
    return True


def main():
    """Main setup function"""
    logger.info("🚀 Setting up AnyLoc Similarity Service...")
    
    # Test imports
    if not test_imports():
        logger.error("❌ Required dependencies missing")
        return False
    
    # Setup directories
    setup_directories()
    
    # Check AnyLoc submodule
    if not check_anyloc_submodule():
        logger.warning("⚠️  AnyLoc submodule issues detected")
        logger.warning("⚠️  Service may not work properly without AnyLoc")
    
    # Setup VLAD vocabulary
    if not setup_vlad_vocabulary():
        logger.error("❌ Failed to setup VLAD vocabulary")
        return False
    
    logger.info("✅ Setup complete!")
    logger.info("")
    logger.info("🎯 Next steps:")
    logger.info("1. Start the service: python scripts/dev_server.py")
    logger.info("2. Visit API docs: http://localhost:8000/docs")
    logger.info("3. Upload images and test similarity search")
    logger.info("")
    logger.info("💡 For production:")
    logger.info("- Download official VLAD centers from AnyLoc repository")
    logger.info("- Use Docker deployment: docker-compose up --build")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
