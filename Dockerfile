# ============================================================
# Superpoint Transformer + SegmentedForests
# ============================================================
# Base:     nvidia/cuda:12.1.1-devel-ubuntu22.04
# Python:   3.10
# PyTorch:  2.2.0 + CUDA 12.1
# Matches the production environment on Ubuntu 24.04 / RTX 4080
# ============================================================

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# ── System packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 / pip the defaults
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip setuptools wheel

# ── Working directory ─────────────────────────────────────────
WORKDIR /workspace/superpoint_transformer

# ── Core ML stack (install first for Docker layer caching) ────
RUN pip install torch==2.2.0 torchvision==0.17.0 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install torchmetrics==0.11.4

RUN pip install \
    pyg_lib==0.4.0+pt22cu121 \
    torch_scatter==2.1.2+pt22cu121 \
    torch_cluster==1.6.3+pt22cu121 \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

RUN pip install torch_geometric==2.3.0

# ── SPT Python dependencies ───────────────────────────────────
RUN pip install \
    matplotlib \
    "plotly==5.9.0" \
    "jupyterlab>=3" \
    "ipywidgets>=7.6" \
    jupyter-dash \
    "notebook>=5.3" \
    ipykernel \
    nb_conda_kernels \
    plyfile==0.9 \
    h5py==3.16.0 \
    colorhash==2.3.0 \
    seaborn==0.13.2 \
    numba==0.65.1 \
    pytorch-lightning==2.6.5 \
    pyrootutils==1.0.4 \
    "hydra-core>=1.3.3" \
    hydra-colorlog==1.2.0 \
    hydra-submitit-launcher==1.2.0 \
    "rich<=14.0" \
    torch_tb_profiler \
    wandb==0.27.2 \
    open3d==0.19.0 \
    gdown==6.1.0 \
    ipyfilechooser==0.6.0 \
    torch-ransac3d==2.0.0 \
    pgeof==0.3.4 \
    pycut-pursuit==0.1.4 \
    pygrid-graph==0.0.4 \
    torch-graph-components==0.1.1

# ── LAZ support (laspy + Rust lazrs backend) ──────────────────
RUN pip install "laspy[lazrs]==2.7.0"

# ── Copy repository code (data/ excluded via .dockerignore) ───
COPY . .

# ── Compile FRNN (Fast Radius Nearest Neighbours) ─────────────
# FRNN requires CUDA dev tools — must be built inside the devel image.
RUN cd src/dependencies/FRNN/external/prefix_sum \
 && pip install . \
 && cd ../.. \
 && pip install .

# ── Environment variables ─────────────────────────────────────
ENV PYTHONPATH=/workspace/superpoint_transformer
ENV TORCH_HOME=/workspace/.cache/torch
ENV WANDB_DIR=/workspace/logs/wandb

# ── Default command ───────────────────────────────────────────
CMD ["bash"]

# ============================================================
# Usage
# ============================================================
#
# Build:
#   docker build -t spt-forest .
#
# Run (interactive, GPU, mount data and logs):
#   docker run --gpus all -it --rm \
#     -v /home/adminlms/superpoint_transformer/data:/workspace/superpoint_transformer/data \
#     -v /home/adminlms/superpoint_transformer/logs:/workspace/superpoint_transformer/logs \
#     spt-forest
#
# Mini preprocessing test (inside the container):
#   python src/train.py datamodule=semantic/forest model=semantic/spt-2 \
#     trainer=gpu +datamodule.mini=True
#
# Full training:
#   python src/train.py datamodule=semantic/forest model=semantic/spt-2 trainer=gpu
#
# Evaluation:
#   python src/eval.py datamodule=semantic/forest model=semantic/spt-2 \
#     trainer=gpu ckpt_path=/workspace/superpoint_transformer/logs/<run>/checkpoints/best.ckpt
# ============================================================
