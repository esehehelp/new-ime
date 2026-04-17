FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Clone repo and install deps
WORKDIR /workspace
RUN git clone https://github.com/esehehelp/new-ime.git && \
    cd new-ime && \
    uv sync --group dev

# Pre-download dataset from HuggingFace
RUN cd new-ime && \
    mkdir -p datasets/eval_v3 && \
    uv run hf download esehe/new-ime-dataset train.jsonl --repo-type dataset --local-dir datasets/eval_v3 && \
    uv run hf download esehe/new-ime-dataset dev.jsonl --repo-type dataset --local-dir datasets/eval_v3 && \
    uv run hf download esehe/new-ime-dataset chunks_v3_100m.jsonl --repo-type dataset --local-dir datasets/eval_v3

WORKDIR /workspace/new-ime

# Default: show help
CMD ["bash", "-c", "echo 'new-ime training container ready. Use scripts in scripts/ to train.' && bash"]
