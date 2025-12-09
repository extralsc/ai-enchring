#!/bin/bash
# =============================================================================
# Product Enrichment Pipeline - Quick Run Script
# =============================================================================
#
# Usage:
#   ./run.sh beta              # Run beta.py (embedding only)
#   ./run.sh beta2             # Run beta2.py (RAG + LLM)
#   ./run.sh app               # Run app.py (full pipeline)
#   ./run.sh shell             # Interactive shell
#   ./run.sh build             # Build Docker image
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Create one from the example:"
    echo "  cp .env.example .env"
    echo "  nano .env  # Add your DATABASE_URL"
    exit 1
fi

# Check for inputs directory
if [ ! -d inputs ]; then
    mkdir -p inputs
    echo -e "${YELLOW}Created inputs/ directory. Add your CSV files there.${NC}"
fi

# Check for outputs directory
if [ ! -d outputs ]; then
    mkdir -p outputs
fi

case "$1" in
    build)
        echo -e "${GREEN}Building Docker image...${NC}"
        docker compose build
        ;;
    beta)
        echo -e "${GREEN}Running beta.py (Embedding classification)...${NC}"
        docker compose run --rm app python beta.py "${@:2}"
        ;;
    beta2)
        echo -e "${GREEN}Running beta2.py (RAG + LLM classification)...${NC}"
        docker compose run --rm app python beta2.py "${@:2}"
        ;;
    beta3)
        echo -e "${GREEN}Running beta3.py (Embedding + Cross-encoder re-ranking)...${NC}"
        docker compose run --rm app python beta3.py "${@:2}"
        ;;
    fashion)
        echo -e "${GREEN}Running fashion_beta.py (FashionSigLIP classification)...${NC}"
        docker compose run --rm app python fashion_beta.py "${@:2}"
        ;;
    fashion-test)
        echo -e "${GREEN}Running fashion_clip_test.py (Test multiple fashion models)...${NC}"
        docker compose run --rm app python fashion_clip_test.py "${@:2}"
        ;;
    app)
        echo -e "${GREEN}Running app.py (Full pipeline)...${NC}"
        docker compose run --rm app python app.py "${@:2}"
        ;;
    shell)
        echo -e "${GREEN}Starting interactive shell...${NC}"
        docker compose run --rm app bash
        ;;
    *)
        echo "Product Enrichment Pipeline"
        echo ""
        echo "Usage: ./run.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  build        Build Docker image"
        echo "  beta         Run embedding-only classification (BGE-M3, fast)"
        echo "  beta2        Run RAG + LLM classification (most accurate, slow)"
        echo "  beta3        Run embedding + cross-encoder re-ranking (accurate, medium)"
        echo "  fashion      Run FashionSigLIP classification (fashion-optimized)"
        echo "  fashion-test Test multiple fashion models (compare accuracy)"
        echo "  app          Run full pipeline"
        echo "  shell        Start interactive shell"
        echo ""
        echo "Examples:"
        echo "  ./run.sh build"
        echo "  ./run.sh beta --input inputs/products.csv"
        echo "  ./run.sh beta2 --top-k 5"
        echo "  ./run.sh beta3 --top-k 10"
        echo "  ./run.sh fashion --input inputs/products.csv"
        echo "  ./run.sh fashion-test --all --limit 1000  # Test all fashion models"
        echo "  ./run.sh shell"
        ;;
esac
