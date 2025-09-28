FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates python3 python3-pip python3-venv ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md ./
RUN pip3 install -U pip && pip3 install -e .

# Optional: preinstall DVC S3 deps
RUN pip3 install "dvc[s3]" pre-commit

ENV PYTHONPATH=/workspace/src
COPY . .
CMD ["bash"]
