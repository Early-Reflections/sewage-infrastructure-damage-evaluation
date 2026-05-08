# DGX Spark (GB10 Grace Blackwell, ARM64 aarch64, CUDA 13) compatible image.
# The NGC PyTorch container ships with Blackwell-optimised PyTorch and Python
# compiled for aarch64 — do NOT use pytorch.org wheel installs (x86_64 only).
FROM nvcr.io/nvidia/pytorch:26.04-py3

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy only the dependency spec first for better layer caching.
COPY pyproject.toml .

# Install in editable mode so the source tree is importable without extra setup.
# --no-build-isolation ensures we use the container's pre-installed torch/CUDA.
RUN pip install --no-cache-dir --no-build-isolation \
        "transformers>=4.40.0" \
        "accelerate>=0.30.0" \
        "datasets>=2.19.0" \
        "pycocotools>=2.0.7" \
        "scikit-image>=0.22.0" \
        "timm>=0.9.16" \
        "omegaconf>=2.3.0" \
        "einops>=0.7.0" \
        "wandb>=0.17.0" \
        "gdown>=5.1.0" \
        "pillow>=10.3.0" \
        "numpy>=1.26.0" \
        "scipy>=1.13.0" \
        "tqdm>=4.66.0" \
        "pandas>=2.2.0"

# ── Project source ─────────────────────────────────────────────────────────────
COPY . .
RUN pip install --no-cache-dir --no-build-isolation -e .

ENV PYTHONPATH=/workspace
ENV TOKENIZERS_PARALLELISM=false

CMD ["bash"]
