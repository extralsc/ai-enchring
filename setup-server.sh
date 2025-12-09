#!/bin/bash
# =============================================================================
# GPU Server Setup Script
# =============================================================================
# Run this on a fresh Ubuntu 22.04 GPU server to install everything needed.
#
# Usage:
#   chmod +x setup-server.sh
#   ./setup-server.sh
#
# After running, REBOOT the server, then run:
#   ./run.sh build
#   ./run.sh beta
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GPU Server Setup Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Please run without sudo (script will ask for sudo when needed)${NC}"
    exit 1
fi

# =============================================================================
# Step 1: Update system
# =============================================================================
echo -e "\n${YELLOW}[1/5] Updating system...${NC}"
sudo apt update && sudo apt upgrade -y

# =============================================================================
# Step 2: Install NVIDIA drivers
# =============================================================================
echo -e "\n${YELLOW}[2/5] Installing NVIDIA drivers...${NC}"

# Check if NVIDIA driver is already installed
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA driver already installed:${NC}"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    sudo apt install -y nvidia-driver-535
    echo -e "${YELLOW}NVIDIA driver installed. REBOOT REQUIRED after setup!${NC}"
fi

# =============================================================================
# Step 3: Install Docker
# =============================================================================
echo -e "\n${YELLOW}[3/5] Installing Docker...${NC}"

if command -v docker &> /dev/null; then
    echo -e "${GREEN}Docker already installed: $(docker --version)${NC}"
else
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}Docker installed${NC}"
fi

# =============================================================================
# Step 4: Install NVIDIA Container Toolkit
# =============================================================================
echo -e "\n${YELLOW}[4/5] Installing NVIDIA Container Toolkit...${NC}"

if dpkg -l | grep -q nvidia-container-toolkit; then
    echo -e "${GREEN}NVIDIA Container Toolkit already installed${NC}"
else
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    sudo systemctl restart docker
    echo -e "${GREEN}NVIDIA Container Toolkit installed${NC}"
fi

# =============================================================================
# Step 5: Install Docker Compose & utilities
# =============================================================================
echo -e "\n${YELLOW}[5/5] Installing Docker Compose & utilities...${NC}"

sudo apt install -y docker-compose-plugin nvtop htop

# =============================================================================
# Create directories
# =============================================================================
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p inputs outputs

# =============================================================================
# Check .env file
# =============================================================================
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${YELLOW}Created .env from .env.example${NC}"
        echo -e "${YELLOW}Please edit .env and add your DATABASE_URL${NC}"
    fi
fi

# =============================================================================
# Make scripts executable
# =============================================================================
chmod +x run.sh 2>/dev/null || true

# =============================================================================
# Done
# =============================================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT: Reboot the server now:${NC}"
echo "  sudo reboot"
echo ""
echo -e "${YELLOW}After reboot, verify GPU:${NC}"
echo "  nvidia-smi"
echo ""
echo -e "${YELLOW}Then run:${NC}"
echo "  # Edit .env with your DATABASE_URL"
echo "  nano .env"
echo ""
echo "  # Build Docker image"
echo "  ./run.sh build"
echo ""
echo "  # Run classification"
echo "  ./run.sh beta"
echo ""
