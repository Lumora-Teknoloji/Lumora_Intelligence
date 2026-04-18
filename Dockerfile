FROM python:3.12-slim

WORKDIR /app

# Sistem bağımlılıklarını yükle (Catboost vb. derleme gerektirebilir)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

EXPOSE 8001

CMD ["python", "run_server.py"]
