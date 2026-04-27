FROM python:3.11-slim

# Dependencias del sistema para ultralytics / cv2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY UBICACION_MANGA.pt .
COPY PANORAMICA_FIGURA_8.pt .

# Render asigna el puerto via $PORT (default 10000)
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
