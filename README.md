# AnyLoc Similarity Service

A production-ready FastAPI backend service for universal visual place recognition and image similarity search using AnyLoc (IEEE RA-L 2023). The system accepts image uploads, extracts AnyLoc features (DINOv2 + VLAD), stores them in SQLite database with chunked feature storage, and provides similarity search capabilities.

## Features

- **Universal Visual Place Recognition**: Built on AnyLoc research (IEEE RA-L 2023)
- **High-Performance Feature Extraction**: DINOv2 ViT-G/14 + VLAD aggregation (98,304D vectors)
- **Scalable Storage**: SQLite with chunked numpy feature storage
- **Fast Similarity Search**: Cosine similarity with sub-100ms search times
- **Production Ready**: Docker deployment with GPU support
- **RESTful API**: FastAPI with automatic documentation

## Technology Stack

- **Backend**: FastAPI with SQLite database (Python 3.9+)
- **AI Models**: AnyLoc (DINOv2 ViT-G/14 + VLAD aggregation)
- **Database**: SQLite with chunked numpy feature storage
- **Containerization**: Docker with NVIDIA GPU support
- **Image Processing**: PIL, OpenCV, torchvision
- **Vector Operations**: NumPy for cosine similarity search

## Quick Start

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)
- Docker and Docker Compose (for containerized deployment)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd anyloc-similarity-service
```

2. **Initialize submodules**:
```bash
git submodule update --init --recursive
```

3. **Set up the environment**:
```bash
chmod +x setup.sh
./setup.sh
```

4. **Start the service**:
```bash
chmod +x start.sh
./start.sh
```

### Docker Deployment

```bash
cd backend
docker-compose up --build
```

## API Documentation

Once the service is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main Endpoints

- **Health Check**: `GET /api/v1/health`
- **Upload Image**: `POST /api/v1/upload`
- **Find Similar**: `POST /api/v1/similarity/{file_id}?top_k=5`
- **Database Stats**: `GET /api/v1/database/stats`
- **Search by Metadata**: `GET /api/v1/search?filename=X&upload_after=Y`
- **Delete Image**: `DELETE /api/v1/uploads/{file_id}`

## Architecture

### Image Processing Pipeline
1. **Upload**: Receive multipart/form-data image file
2. **Validation**: Check file type, size, format
3. **Storage**: Save to uploads/{file_id}.{ext}
4. **Preprocessing**: PIL load → RGB → 224×224 → normalize → 14×14 patches
5. **Feature Extraction**: DINOv2 forward pass → layer 31 value facet
6. **VLAD Aggregation**: Cluster assignment → residual computation → L2 norm
7. **Database Storage**: Normalize features → add to chunk → update SQLite
8. **Search**: Cosine similarity against all database features

### Database Schema
- **images**: Main table with metadata and feature pointers
- **feature_chunks**: Tracking for chunked numpy feature storage
- **search_history**: Analytics for search operations

### Feature Storage
- Features stored in chunked numpy arrays: `features/chunk_N.npy`
- Each chunk: 1000 images × 98,304 dimensions
- Cosine similarity for search
- Async operations with ThreadPoolExecutor

## Performance

- **Feature Extraction**: ~1-2 seconds per image on GPU
- **Similarity Search**: <100ms for 10K images database
- **Concurrent Requests**: Multiple simultaneous uploads/searches
- **Storage Efficiency**: Chunked feature storage for memory management

## Configuration

The service can be configured via environment variables:

```bash
# API Configuration
DEBUG=false
HOST=0.0.0.0
PORT=8000

# File Handling
MAX_FILE_SIZE=10485760  # 10MB
UPLOAD_DIR=./uploads

# AnyLoc Settings
DEVICE=auto
DINO_MODEL=dinov2_vitg14
DESC_LAYER=31
DESC_FACET=value
VLAD_CLUSTERS=64
VLAD_DESC_DIM=1536
```

## Development

### Local Development
```bash
cd backend
pip install -r requirements.txt
python scripts/dev_server.py
```

### Testing
```bash
cd backend
pytest tests/
```

### Setup Models
```bash
cd backend
python scripts/setup_models.py
```

## Project Structure

```
anyloc-similarity-service/
├── backend/
│   ├── app/
│   │   ├── api/              # API endpoints and models
│   │   ├── core/             # Core business logic
│   │   └── utils/            # Utility functions
│   ├── anyloc/               # Git submodule
│   ├── models/               # VLAD vocabularies and database
│   ├── uploads/              # Uploaded images
│   ├── tests/                # Test suite
│   ├── scripts/              # Setup and utility scripts
│   └── Dockerfile
├── frontend/                 # Placeholder for future UI
└── docs/                     # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **AnyLoc**: Built on the AnyLoc research from IEEE RA-L 2023
- **DINOv2**: Uses Meta's DINOv2 vision transformer
- **VLAD**: Vector of Locally Aggregated Descriptors for feature aggregation

## Troubleshooting

### Common Issues

1. **CUDA not available**: Ensure NVIDIA drivers and CUDA toolkit are installed
2. **Out of memory**: Reduce batch size or use CPU mode
3. **Model loading fails**: Run `python scripts/setup_models.py` to initialize vocabularies
4. **Permission denied**: Ensure proper file permissions on uploads directory

For more detailed troubleshooting, check the logs in the `logs/` directory.
