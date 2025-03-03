# PowerShell script for Windows using 7-Zip for faster extraction

# Path to 7-Zip executable (default installation path)
$7zPath = "C:\Program Files\7-Zip\7z.exe"

# Check if 7-Zip exists
if (-not (Test-Path -Path $7zPath)) {
    Write-Host "7-Zip not found at $7zPath" -ForegroundColor Red
    Write-Host "Please install 7-Zip from https://www.7-zip.org/ or update the script with the correct path" -ForegroundColor Yellow
    exit 1
}

# Activate conda environment
Write-Host "Activating pytorch_env conda environment..." -ForegroundColor Cyan
conda activate pytorch_env

# Create data directory if it doesn't exist
if (-not (Test-Path -Path ".\data")) {
    Write-Host "Creating data directory..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path ".\data" | Out-Null
}

# Download the dataset
Write-Host "Downloading dataset from Kaggle..." -ForegroundColor Cyan
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p .\data

# Get the downloaded zip file
$zipFile = Get-ChildItem -Path ".\data" -Filter "*.zip" | Select-Object -First 1

if ($null -eq $zipFile) {
    Write-Host "No zip file found in .\data directory. Download may have failed." -ForegroundColor Red
    exit 1
}

Write-Host "Extracting dataset using 7-Zip (fast extraction)..." -ForegroundColor Cyan
# Extract to data directory
& "$7zPath" x "$($zipFile.FullName)" "-o.\data" -y

# Remove the zip file
Write-Host "Cleaning up..." -ForegroundColor Cyan
Remove-Item -Path $zipFile.FullName

Write-Host "Setup complete! Dataset is ready in .\data" -ForegroundColor Green

# Run data augmentation
Write-Host "Running data augmentation..." -ForegroundColor Cyan
python utils/augmentation.py --input ./data/Train --output ./data/Train_Augmented --num-squares 8 --min-size 20 --max-size 60

# Run PCA dimensionality reduction
Write-Host "Performing PCA dimensionality reduction..." -ForegroundColor Cyan
python utils/pca.py --n-components 150 --batch-size 512

Write-Host "All processing complete!" -ForegroundColor Green