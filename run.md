# Product Enrichment Pipeline - Docker Guide

## GPU Server Setup (Fresh Ubuntu Server)

### Quick Setup (Automated)

```bash
# Upload/clone your code to the server, then:
cd ai-test
chmod +x setup-server.sh
./setup-server.sh

# Reboot after setup
sudo reboot

# After reboot, verify GPU works
nvidia-smi

# Edit .env with your database URL
nano .env

# Build and run
./run.sh build
./run.sh beta
```

### Manual Setup

#### 1. Install NVIDIA Drivers

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535

# Reboot
sudo reboot

# Verify GPU (after reboot)
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   30C    P0    50W / 400W |      0MiB / 40960MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

#### 2. Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (no sudo needed)
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker
docker --version
```

#### 3. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Verify GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### 4. Install Docker Compose

```bash
# Install Docker Compose
sudo apt install -y docker-compose-plugin

# Verify
docker compose version
```

## Monitoring GPU

### Check GPU Status
```bash
# One-time check
nvidia-smi

# Live monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi -q
```

### Monitor During Training/Inference
```bash
# In a separate terminal, run:
watch -n 1 nvidia-smi

# Or use nvtop (prettier)
sudo apt install -y nvtop
nvtop
```

### Check GPU Memory Usage
```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 1
```

### Check Running Processes on GPU
```bash
nvidia-smi pmon -i 0
```

---

## Prerequisites

- Docker with NVIDIA GPU support
- GPU server (A100, A10, RTX 4090, etc.)
- PostgreSQL database with categories

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo>
cd ai-test

# Create environment file
cp .env.example .env

# Edit with your database credentials
nano .env
```

**.env file:**
```
DATABASE_URL=postgresql://user:password@host:5432/database
```

### 2. Prepare Data

```bash
# Create directories
mkdir -p inputs outputs

# Add your product CSV to inputs/
cp /path/to/products.csv inputs/
```

**CSV format** (semicolon-separated, iso-8859-1 encoding):
```csv
product_type;title;description;...
Jacket;Winter Jacket Pro;Warm waterproof jacket...;...
```

### 3. Build Docker Image

```bash
./run.sh build

# Or manually:
docker-compose build
```

### 4. Run Classification

**Option A: Embedding Only (Fast)**
```bash
./run.sh beta
```
- Speed: ~500K products in 2-3 minutes
- Output: `outputs/products_beta.csv`

**Option B: RAG + LLM (Accurate)**
```bash
./run.sh beta2
```
- Speed: ~500K products in 30-60 minutes
- Output: `outputs/products_beta2.csv`

**Option C: Full Pipeline (app.py)**
```bash
./run.sh app
```
- Includes: colors, categories, gender, size type
- Output: `outputs/enriched_products.csv`

## Commands Reference

| Command | Description |
|---------|-------------|
| `./run.sh build` | Build Docker image |
| `./run.sh beta` | Embedding-only classification |
| `./run.sh beta2` | RAG + LLM classification |
| `./run.sh app` | Full enrichment pipeline |
| `./run.sh shell` | Interactive bash shell |

## Advanced Options

### Custom Input File
```bash
./run.sh beta --input inputs/myfile.csv
./run.sh beta2 --input inputs/myfile.csv
```

### Custom Output File
```bash
./run.sh beta --output outputs/custom_output.csv
```

### Adjust Batch Size
```bash
./run.sh beta --batch-size 512
./run.sh beta2 --batch-size 128
```

### RAG Top-K Candidates
```bash
./run.sh beta2 --top-k 10  # Send 10 candidates to LLM (default: 5)
```

### Use Different Models
```bash
# Smaller/faster embedding model
./run.sh beta --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Different LLM
./run.sh beta2 --llm-model Qwen/Qwen2.5-3B-Instruct

# Disable vLLM (use transformers)
./run.sh beta2 --no-vllm
```

## Docker Commands (Manual)

```bash
# Build
docker-compose build

# Run beta.py
docker-compose run --rm app python beta.py

# Run beta2.py with options
docker-compose run --rm app python beta2.py --top-k 5 --batch-size 256

# Run app.py
docker-compose run --rm app python app.py

# Interactive shell
docker-compose run --rm app bash

# Check GPU
docker-compose run --rm app nvidia-smi
```

## Output Files

| Script | Output File | Added Columns |
|--------|-------------|---------------|
| beta.py | `*_beta.csv` | `predicted_lokalt_kategori`, `predicted_lokalt_kategori_id`, `embedding_confidence` |
| beta2.py | `*_beta2.csv` | `predicted_lokalt_kategori`, `predicted_lokalt_kategori_id`, `rag_confidence`, `top_candidates` |
| app.py | `enriched_*.csv` | colors, categories, gender, size_type, etc. |

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Database Connection Failed
```bash
# Test connection from container
docker-compose run --rm app python -c "
import asyncio
import asyncpg
async def test():
    conn = await asyncpg.connect('postgresql://user:pass@host:5432/db')
    print('Connected!')
    await conn.close()
asyncio.run(test())
"
```

### Out of Memory
```bash
# Reduce batch size
./run.sh beta --batch-size 256
./run.sh beta2 --batch-size 64

# Use smaller models
./run.sh beta2 --llm-model Qwen/Qwen2.5-3B-Instruct
```

### Models Not Downloading
```bash
# Check internet connection
docker-compose run --rm app curl -I https://huggingface.co

# Manual download inside container
docker-compose run --rm app python -c "
from transformers import AutoModel
AutoModel.from_pretrained('intfloat/multilingual-e5-large')
"
```

## vLLM vs Transformers

vLLM is a fast LLM inference engine. It matters for **beta2.py** and **app.py** (which use LLM).

| | vLLM | Transformers |
|---|------|--------------|
| Speed | **2-4x faster** | Slower |
| Memory | Efficient (PagedAttention) | Uses more VRAM |
| Batching | Smart parallel batching | Sequential |
| 500K products (beta2) | ~15-30 min | ~60-120 min |

**vLLM is installed by default in Docker.** To disable:
```bash
./run.sh beta2 --no-vllm
```

**Note:** beta.py only uses embeddings (no LLM), so vLLM doesn't affect it.

## Performance Comparison

| Script | Method | 500K Products | Accuracy |
|--------|--------|---------------|----------|
| beta.py | Embedding only | ~2-3 min | Good |
| beta2.py | RAG + LLM (vLLM) | ~15-30 min | Best |
| beta2.py | RAG + LLM (no vLLM) | ~60-120 min | Best |
| app.py | Full pipeline | ~5-10 min | Good + extras |

## Recommended Workflow

1. **Test with small dataset first:**
   ```bash
   head -100 inputs/products.csv > inputs/test.csv
   ./run.sh beta --input inputs/test.csv
   ```

2. **Check results:**
   ```bash
   head outputs/test_beta.csv
   ```

3. **Run full dataset:**
   ```bash
   ./run.sh beta  # or beta2 for better accuracy
   ```
