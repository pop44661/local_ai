FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---- system deps ----
RUN apt-get update && apt-get install -y \
    python3 python3-pip git git-lfs ca-certificates curl \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# ---- PyTorch 2.9.1 + CUDA 12.8 ----
RUN pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 \
    && rm -rf /root/.cache/pip

RUN pip install --upgrade pip setuptools wheel packaging cmake ninja psutil
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.9-cp310-cp310-linux_x86_64.whl

# ---- Hugging Face + FastAPI ----
RUN pip install \
    transformers==4.44.2 \
    accelerate==0.32.1 \
    sentence-transformers==2.6.1 \
    huggingface-hub \
    safetensors \
    optimum \
    fastapi \
    auto-gptq \
    uvicorn[standard] \
    && rm -rf /root/.cache/pip

WORKDIR /app

COPY app /app/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]