FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar modelos EasyOCR en el build
RUN python -c "import easyocr; easyocr.Reader(['es', 'en'], gpu=False, verbose=False)"

COPY app.py .
COPY nodos.json .

# ── Mangas estándar ───────────────────────────────────────────────────────────
COPY MANGA_SELLADA_V2.pt .
COPY MANGA_ETIQUETAS_FO_INGRESO_v2.pt .
COPY ETIQUETA_TAPA_MANGA_v2.pt .
COPY UBICACION_MANGA_v2.1.pt .
COPY PANORAMICA_FIGURA_8_V2.pt .
COPY CASETERA_MANGA.pt .
COPY PANORAMICA_MANGA_DESTAPADA.pt .

# ── Mangas 2 hilos ────────────────────────────────────────────────────────────
COPY 2H_MANGA_SELLADA.pt .
COPY 2H_ETIQUETA_TAPA_MANGA.pt .
COPY 2H_UBICACION_MANGA.pt .
COPY 2H_PANORAMICA_FIGURA_8.pt .

# ── ODF ───────────────────────────────────────────────────────────────────────
COPY CASETERA_ODF_v1.pt .
COPY CASETERAS_COMPLETAS_v1.pt .
COPY INGRESO_FO_AL_ODF_v1.pt .
COPY PANORAMICA_FRONTAL_ODF_v1.pt .
COPY PANORAMICA_POSTERIOR_ODF_v1.pt .

# Render asigna el puerto via $PORT (default 10000)
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
