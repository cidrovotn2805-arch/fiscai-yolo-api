FROM python:3.11-slim

# Dependencias del sistema para ultralytics / cv2 / easyocr
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar modelos EasyOCR en el build (evita descarga en runtime)
RUN python -c "import easyocr; easyocr.Reader(['es', 'en'], gpu=False, verbose=False)"

COPY app.py .
COPY MANGA_SELLADA.pt .
COPY ETIQUETAS_FO_INGRESO.pt .
COPY ETIQUETA_TAPA_MANGA.pt .
COPY UBICACION_MANGA.pt .
COPY PANORAMICA_FIGURA_8.pt .
COPY 2H_MANGA_SELLADA.pt .
COPY 2H_ETIQUETA_TAPA_MANGA.pt .
COPY 2H_UBICACION_MANGA.pt .
COPY 2H_PANORAMICA_FIGURA_8.pt .

# Render asigna el puerto via $PORT (default 10000)
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
