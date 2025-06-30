# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  python3-dev \
  git \
  wget \
  curl \
  build-essential \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libgl1-mesa-glx \
  && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
  pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Initialize git submodules (if not done already)
RUN git submodule update --init --recursive || true

# Create necessary directories
RUN mkdir -p uploads models/vocabulary models/features logs

# Set Python path
ENV PYTHONPATH=/app:/app/anyloc

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
