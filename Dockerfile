# 1. Senin yerelindeki sürümle birebir aynı Python imajı
FROM python:3.11-slim

# 2. Sistem bağımlılıklarını kuruyoruz (Prefect ve MLflow için gerekli araçlar)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Çalışma dizini
WORKDIR /app

# 4. Bağımlılıkları kopyala ve kur
# (requirements.txt dosyanın kök dizinde olduğunu varsayıyorum)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Proje klasörlerini kopyala
COPY . .

# 6. Prefect ve MLflow için default portları açık bırakalım
EXPOSE 4200 5000 8000

# Şimdilik bir komut vermiyoruz, Dev Containers veya Compose ile yöneteceğiz
