# Git Setup and Push Script
# This script initializes git, creates a new branch, and pushes to GitHub

Write-Host "AlloyTower Real Estate ML - Git Setup" -ForegroundColor Cyan
Write-Host "=" * 60

# Check if git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "`nInitializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✓ Git initialized" -ForegroundColor Green
} else {
    Write-Host "`n✓ Git repository already initialized" -ForegroundColor Green
}

# Add all files
Write-Host "`nStaging files..." -ForegroundColor Yellow
git add .
Write-Host "✓ Files staged" -ForegroundColor Green

# Create commit
Write-Host "`nCreating commit..." -ForegroundColor Yellow
$commitMessage = "feat: Add ML models, documentation, and clean code structure

- Implemented ExtraTrees model (MAE: 36.63 days, R²: 0.584)
- Added comprehensive model documentation
- Organized code following Google Python style guide
- Added future date prediction feature
- Created detailed documentation in docs/ folder
- Cleaned up repository structure
- Added proper .gitignore
"

git commit -m $commitMessage
Write-Host "✓ Commit created" -ForegroundColor Green

# Create new branch
$branchName = "feature/ml-models-and-documentation"
Write-Host "`nCreating branch: $branchName..." -ForegroundColor Yellow
git checkout -b $branchName
Write-Host "✓ Branch created and checked out" -ForegroundColor Green

# Show status
Write-Host "`nGit Status:" -ForegroundColor Cyan
git status

Write-Host "`n" + "=" * 60
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Add remote if not already added:" -ForegroundColor Yellow
Write-Host "   git remote add origin https://github.com/yourusername/alloy-tower-real-estate-ml.git"
Write-Host "`n2. Push to GitHub:" -ForegroundColor Yellow
Write-Host "   git push -u origin $branchName"
Write-Host "`n3. Create Pull Request on GitHub" -ForegroundColor Yellow
Write-Host "=" * 60
