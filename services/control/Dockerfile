FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    docker.io \
    libcurl4-openssl-dev \
    libssl-dev \
    build-essential \
    docker-compose \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

ARG BUILD_DIR
ENV BUILD_DIR=${BUILD_DIR}

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]