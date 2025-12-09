#!/bin/bash
# Setup script for Product Enrichment Pipeline
# Run this on your A10 GPU server

set -e

echo "=== Product Enrichment Pipeline Setup ==="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

echo "Python: $(python --version)"
echo "Pip: $(pip --version)"

# Uninstall any broken torch first
echo "Removing any existing torch installation..."
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Install PyTorch with CUDA 11.8 (standard for AWS A10)
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
echo "Verifying CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install other dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install vLLM for fast LLM inference
echo "Installing vLLM..."
pip install vllm

# Verify installation
echo ""
echo "=== Verifying Installation ==="
python -c "
import torch
import transformers
import sentence_transformers
import open_clip
print('All imports successful!')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else '')
"

# Try to import vllm
python -c "from vllm import LLM; print('vLLM: OK')" 2>/dev/null || echo "vLLM: Not available (will fallback to transformers)"

echo ""
echo "=== Setup Complete ==="
echo "Run the pipeline with: python app.py"
