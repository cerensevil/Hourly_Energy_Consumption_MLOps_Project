# 1️⃣ Python image
FROM python:3.11-slim

# 2️⃣ Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 3️⃣ Çalışma dizini
WORKDIR /app

# 4️⃣ Requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Proje dosyaları
COPY . .

# 6️⃣ Port
EXPOSE 8000

# 7️⃣ API başlat
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
