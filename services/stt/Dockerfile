FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y git ffmpeg software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.12 python3-distutils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.4.11 /uv /bin/uv

WORKDIR /root

RUN git clone --branch v0.5.0 https://github.com/speaches-ai/speaches.git \
    && mv speaches faster-whisper-server

WORKDIR /root/faster-whisper-server

RUN find . -mindepth 1 -maxdepth 1 ! -name src ! -name examples -exec rm -rf {} +

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --refresh --extra ui

ENV WHISPER__MODEL=Systran/faster-distil-whisper-large-v3
ENV WHISPER__INFERENCE_DEVICE=auto
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000

CMD ["uv", "run", "uvicorn", "--factory", "faster_whisper_server.main:create_app"]

