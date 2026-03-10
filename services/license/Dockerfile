# openssl genrsa -out private.pem 2048
# openssl rsa -in private.pem -pubout -out public.pem

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY public.pem .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]