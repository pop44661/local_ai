FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-distutils \
    python3-pkg-resources \
    build-essential \
    git \
    ffmpeg \
    sox \
    libsox-dev \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /workspace
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
COPY requirements.txt ./CosyVoice/
COPY api.py ./CosyVoice/
RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir --no-build-isolation openai-whisper==20231117
RUN python -m pip install --no-deps --no-cache-dir deepspeed==0.15.1 
RUN python -m pip install --no-cache-dir \
    torch==2.9.1+cu128 \
    torchaudio==2.9.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
RUN python -m pip install --no-cache-dir \
    onnxruntime-gpu==1.18.0 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

RUN python -m pip install --no-cache-dir -r /workspace/CosyVoice/requirements.txt
RUN python -m pip install --no-cache-dir --force-reinstall \
    ruamel.yaml==0.17.32 \
    hyperpyyaml==1.2.2
RUN python -m pip install --no-cache-dir \
    --pre torchcodec==0.9.1

WORKDIR /workspace/CosyVoice
ENV PYTHONPATH=/workspace/CosyVoice
ENV PRELOAD_MODELS=CosyVoice2-0.5B
EXPOSE 8000
CMD ["sh", "-c", "python api.py --host 0.0.0.0 --port 8000 --preload-models $PRELOAD_MODELS"]
