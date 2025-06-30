#!/bin/bash

# AnyLoc Similarity Service Setup Script

set -e

echo "🚀 Setting up AnyLoc Similarity Service..."

# Check if we're in the right directory
if [ ! -f "setup.sh" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Initialize git submodules
echo "📦 Initializing git submodules..."
git submodule update --init --recursive

# Create required directories
echo "📁 Creating directory structure..."
mkdir -p backend/uploads
mkdir -p backend/models/vocabulary
mkdir -p backend/models/features
mkdir -p backend/logs
mkdir -p frontend
mkdir -p docs

# Create .gitkeep files for empty directories
touch backend/uploads/.gitkeep
touch backend/models/.gitkeep
touch frontend/.gitkeep
touch docs/.gitkeep

# Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup models and vocabularies
echo "🤖 Setting up AnyLoc models..."
python scripts/setup_models.py

cd ..

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Start the service: ./start.sh"
echo "2. Visit API documentation: http://localhost:8000/docs"
echo "3. Upload images and test similarity search"
echo ""
echo "💡 For Docker deployment: cd backend && docker-compose up --build"
