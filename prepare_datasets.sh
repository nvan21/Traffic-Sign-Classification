#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for extraction tools (in order of preference)
if command -v 7z &> /dev/null; then
    EXTRACT_CMD="7z"
    echo -e "${CYAN}Using 7z for fast extraction${NC}"
elif command -v unzip &> /dev/null; then
    EXTRACT_CMD="unzip"
    echo -e "${CYAN}Using unzip for extraction${NC}"
else
    echo -e "${RED}No extraction tool found${NC}"
    echo -e "${YELLOW}Please install 7z (p7zip-full) or unzip using your package manager${NC}"
    echo -e "${YELLOW}Example: 'sudo apt-get install p7zip-full' or 'brew install p7zip'${NC}"
    exit 1
fi

# Activate conda environment
echo -e "${CYAN}Activating pytorch_env conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate pytorch_env || { echo -e "${RED}Failed to activate conda environment${NC}"; exit 1; }

# Create data directory if it doesn't exist
if [ ! -d "./data" ]; then
    echo -e "${CYAN}Creating data directory...${NC}"
    mkdir -p ./data
fi

# Download the dataset
echo -e "${CYAN}Downloading dataset from Kaggle...${NC}"
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p ./data || { echo -e "${RED}Failed to download dataset${NC}"; exit 1; }

# Get the downloaded zip file
zipFile=$(find ./data -name "*.zip" -type f | head -n 1)

if [ -z "$zipFile" ]; then
    echo -e "${RED}No zip file found in ./data directory. Download may have failed.${NC}"
    exit 1
fi

echo -e "${CYAN}Extracting dataset using fast extraction...${NC}"
# Extract to data directory based on available tool
if [ "$EXTRACT_CMD" = "7z" ]; then
    7z x "$zipFile" -o./data -y || { echo -e "${RED}Extraction failed${NC}"; exit 1; }
else
    unzip -o "$zipFile" -d ./data || { echo -e "${RED}Extraction failed${NC}"; exit 1; }
fi

# Remove the zip file
echo -e "${CYAN}Cleaning up...${NC}"
rm -f "$zipFile"

echo -e "${GREEN}Setup complete! Dataset is ready in ./data${NC}"

# Run data augmentation
echo -e "${CYAN}Running data augmentation...${NC}"
python utils/augmentation.py --input ./data/Train --output ./data/Train_Augmented --num-squares 8 --min-size 20 --max-size 60 || { echo -e "${RED}Data augmentation failed${NC}"; exit 1; }

# Run PCA dimensionality reduction
echo -e "${CYAN}Performing PCA dimensionality reduction...${NC}"
python utils/pca.py --n-components 150 --batch-size 512 || { echo -e "${RED}PCA dimensionality reduction failed${NC}"; exit 1; }

echo -e "${GREEN}All processing complete!${NC}"