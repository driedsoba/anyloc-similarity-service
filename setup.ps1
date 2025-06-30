# AnyLoc Similarity Service Setup Script for Windows
# PowerShell version

Write-Host "🚀 Setting up AnyLoc Similarity Service..." -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "setup.sh")) {
  Write-Host "❌ Please run this script from the project root directory" -ForegroundColor Red
  exit 1
}

# Initialize git submodules
Write-Host "📦 Initializing git submodules..." -ForegroundColor Yellow
git submodule update --init --recursive

# Create required directories
Write-Host "📁 Creating directory structure..." -ForegroundColor Yellow
$directories = @(
  "backend\uploads",
  "backend\models\vocabulary",
  "backend\models\features",
  "backend\logs",
  "frontend",
  "docs"
)

foreach ($dir in $directories) {
  if (-not (Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "  Created: $dir" -ForegroundColor Gray
  }
}

# Create .gitkeep files for empty directories
$gitkeepFiles = @(
  "backend\uploads\.gitkeep",
  "backend\models\.gitkeep",
  "frontend\.gitkeep",
  "docs\.gitkeep"
)

foreach ($file in $gitkeepFiles) {
  if (-not (Test-Path $file)) {
    New-Item -ItemType File -Path $file -Force | Out-Null
  }
}

# Set up Python virtual environment
Write-Host "🐍 Setting up Python virtual environment..." -ForegroundColor Yellow
Set-Location backend

if (-not (Test-Path "venv")) {
  python -m venv venv
}

# Activate virtual environment
& "venv\Scripts\Activate.ps1"

# Install Python dependencies
Write-Host "📚 Installing Python dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# Setup models and vocabularies
Write-Host "🤖 Setting up AnyLoc models..." -ForegroundColor Yellow
python scripts\setup_models.py

Set-Location ..

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 Next steps:" -ForegroundColor Cyan
Write-Host "1. Start the service: .\start.ps1" -ForegroundColor White
Write-Host "2. Visit API documentation: http://localhost:8000/docs" -ForegroundColor White
Write-Host "3. Upload images and test similarity search" -ForegroundColor White
Write-Host ""
Write-Host "💡 For Docker deployment: cd backend && docker-compose up --build" -ForegroundColor Yellow
