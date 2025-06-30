# AnyLoc Similarity Service Start Script for Windows
# PowerShell version

Write-Host "🚀 Starting AnyLoc Similarity Service..." -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "start.ps1")) {
    Write-Host "❌ Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

Set-Location backend

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "❌ Virtual environment not found. Please run .\setup.ps1 first" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
& "venv\Scripts\Activate.ps1"

# Check if models are set up
if (-not (Test-Path "models\vocabulary\dinov2_vitg14\l31_value_c64\c_centers.pt")) {
    Write-Host "⚠️  Models not found. Setting up..." -ForegroundColor Yellow
    python scripts\setup_models.py
}

# Create logs directory if it doesn't exist
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null
}

# Set environment variables
$env:PYTHONPATH = "$PWD;$PWD\anyloc"

Write-Host "🌐 Starting FastAPI server..." -ForegroundColor Green
Write-Host "📖 API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "🔍 ReDoc: http://localhost:8000/redoc" -ForegroundColor Cyan
Write-Host "💚 Health Check: http://localhost:8000/api/v1/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow

# Start the development server
python scripts\dev_server.py
