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

# Convertir todos los .pt a ONNX y eliminarlos.
# ONNX + onnxruntime necesita ~75% menos RAM que PyTorch (150MB vs 600MB por modelo),
# lo que permite correr en el tier gratuito de Render (512 MB).
# Este layer queda cacheado: rebuilds por cambios de código no re-exportan los modelos.
RUN python -c "
import os, glob
from ultralytics import YOLO
pts = sorted(glob.glob('/app/*.pt'))
print(f'[onnx-export] {len(pts)} modelos a convertir', flush=True)
for pt in pts:
    name = os.path.basename(pt)
    print(f'[onnx-export] {name} ...', flush=True)
    YOLO(pt).export(format='onnx', simplify=True)
    os.remove(pt)
    print(f'[onnx-export] {name.replace(\".pt\",\".onnx\")} listo', flush=True)
print('[onnx-export] Completado.', flush=True)
"

# Render asigna el puerto via $PORT (default 10000)
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
