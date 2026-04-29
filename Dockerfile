FROM python:3.12-slim

WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılıkları kopyala ve yükle
COPY requirements.txt .
# Sadece gerekli paketleri yükle, torch cpu versiyonunu yükleyerek imaj boyutunu küçültebiliriz (eğer CUDA gerekmiyorsa)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Portu expose et
EXPOSE 8001

# Uygulamayı başlat
CMD ["python", "run_server.py"]
