#!/bin/bash

# RKNN Model Inference Setup Script for Vicharak Axon Board
# This script automates the environment setup for running .rknn models

set -e  # Exit on any error

echo "=========================================="
echo "RKNN Toolkit Lite2 Setup for Vicharak Axon"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="env-rknn"
TOOLKIT_REPO="https://github.com/airockchip/rknn-toolkit2.git"
RUNTIME_URL="https://github.com/airockchip/rknn-toolkit2/raw/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"

# Detect Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
echo -e "${YELLOW}Detected Python version: ${PYTHON_VERSION}${NC}"

# Step 1: Install system dependencies
echo -e "\n${GREEN}[1/6] Installing system dependencies...${NC}"
sudo apt update
sudo apt-get install -y python3-dev python3-pip python3-venv gcc wget git
sudo apt install -y python3-opencv python3-numpy python3-setuptools

# Step 2: Create virtual environment
echo -e "\n${GREEN}[2/6] Creating Python virtual environment...${NC}"
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Virtual environment '$VENV_NAME' already exists. Skipping creation.${NC}"
else
    python3 -m venv "$VENV_NAME"
    echo "Virtual environment created: $VENV_NAME"
fi

# Step 3: Activate virtual environment
echo -e "\n${GREEN}[3/6] Activating virtual environment...${NC}"
source "$VENV_NAME/bin/activate"

# Step 4: Clone RKNN Toolkit2 repository
echo -e "\n${GREEN}[4/6] Cloning RKNN Toolkit2 repository...${NC}"
if [ -d "rknn-toolkit2" ]; then
    echo -e "${YELLOW}rknn-toolkit2 directory already exists. Pulling latest changes...${NC}"
    cd rknn-toolkit2 && git pull && cd ..
else
    git clone "$TOOLKIT_REPO"
fi

# Step 5: Install RKNN Toolkit Lite2
echo -e "\n${GREEN}[5/6] Installing RKNN Toolkit Lite2...${NC}"
WHEEL_DIR="rknn-toolkit2/rknn-toolkit-lite2/packages"

# Find the correct wheel file for the Python version
WHEEL_FILE=$(ls "$WHEEL_DIR"/rknn_toolkit_lite2-*-"${PYTHON_VERSION}"-*aarch64*.whl 2>/dev/null | head -n 1)

if [ -z "$WHEEL_FILE" ]; then
    echo -e "${RED}Error: No compatible wheel found for Python ${PYTHON_VERSION}${NC}"
    echo "Available wheels:"
    ls "$WHEEL_DIR"/*.whl
    exit 1
fi

echo "Installing: $WHEEL_FILE"
pip install "$WHEEL_FILE"

# Setup runtime library
echo -e "\n${GREEN}[6/6] Setting up RKNN runtime library...${NC}"
if [ -f "/usr/lib/librknnrt.so" ]; then
    echo -e "${YELLOW}Runtime library already present at /usr/lib/librknnrt.so${NC}"
else
    echo "Downloading librknnrt.so..."
    wget -q "$RUNTIME_URL" -O librknnrt.so
    sudo cp librknnrt.so /usr/lib/
    rm librknnrt.so
    echo "Runtime library placed at /usr/lib/"
fi

# Verify installation
echo -e "\n${GREEN}Verifying installation...${NC}"
python3 -c "from rknnlite.api import RKNNLite; print('RKNN Toolkit Lite2 installed successfully!')"

echo -e "\n${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "To use the environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "Example inference command:"
echo "  python rknn_inference.py --model model.rknn --image input.jpg --size 640 --quantized"
echo ""
