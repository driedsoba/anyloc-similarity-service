#!/bin/bash

# AnyLoc Similarity Service Start Script

set -e

echo "🚀 Starting AnyLoc Similarity Service..."

# Check if we're in the right directory
if [ ! -f "start.sh" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Check if models are set up
if [ ! -f "models/vocabulary/dinov2_vitg14/l31_value_c64/c_centers.pt" ]; then
    echo "⚠️  Models not found. Setting up..."
    python scripts/setup_models.py
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Export environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/anyloc"

echo "🌐 Starting FastAPI server..."
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🔍 ReDoc: http://localhost:8000/redoc"
echo "💚 Health Check: http://localhost:8000/api/v1/health"
echo ""
echo "Press Ctrl+C to stop the server"

# Start the development server
python scripts/dev_server.py
