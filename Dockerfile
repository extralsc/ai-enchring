# =============================================================================
# Product Enrichment Pipeline - GPU Docker Image
# =============================================================================
# Base: NVIDIA CUDA 11.8 + Python 3.11
# Optimized for: A100, A10, RTX 4090, etc.
#
# Build:
#   docker build -t product-enrichment .
#
# Run:
#   docker run --gpus all -v ./inputs:/app/inputs -v ./outputs:/app/outputs \
#       --env-file .env product-enrichment python beta.py
# =============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 11.8 first (large download)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install vLLM for fast LLM inference
RUN pip install vllm

# Copy application code
COPY *.py ./
COPY .env* ./

# Create directories for inputs/outputs
RUN mkdir -p inputs outputs

# Pre-download models (optional - makes container larger but faster startup)
# Uncomment these lines to bake models into the image:
# RUN python -c "from transformers import AutoModel, AutoTokenizer; \
#     AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large'); \
#     AutoModel.from_pretrained('intfloat/multilingual-e5-large')"

# Default command - show help
CMD ["python", "-c", "print('Product Enrichment Pipeline\\n\\nAvailable scripts:\\n  python app.py      - Full pipeline (colors, categories, etc.)\\n  python beta.py     - Embedding-only classification\\n  python beta2.py    - RAG + LLM classification\\n\\nExample:\\n  docker run --gpus all -v ./inputs:/app/inputs -v ./outputs:/app/outputs --env-file .env product-enrichment python beta.py')"]
