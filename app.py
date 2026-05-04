from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from typing import Optional
import base64, io, os, gc, threading, requests as http_requests

app = FastAPI(title="FiscAI YOLO API — Todos los modelos de manga")

# ── Lazy loading: solo 1 modelo en memoria a la vez ──────────────────────────
_model_lock  = threading.Lock()
_model_cache: dict = {}   # { model_key: YOLO }

MODEL_FILES = {
    "manga":           "MANGA_SELLADA.pt",
    "etiquetas":       "ETIQUETAS_FO_INGRESO.pt",
    "etiqueta-tapa":   "ETIQUETA_TAPA_MANGA.pt",
    "ubicacion-manga": "UBICACION_MANGA.pt",
    "panoramica-f8":   "PANORAMICA_FIGURA_8.pt",
}

def _get_model(model_key: str) -> YOLO:
    with _model_lock:
        if model_key not in _model_cache:
            # Liberar todos los modelos cargados para no superar 512 MB
            for k in list(_model_cache.keys()):
                del _model_cache[k]
            gc.collect()
            # Cargar el modelo solicitado desde disco
            path = os.path.join(os.path.dirname(__file__), MODEL_FILES[model_key])
            print(f"[lazy] Cargando modelo: {model_key} ({MODEL_FILES[model_key]})")
            _model_cache[model_key] = YOLO(path)
            print(f"[lazy] Modelo listo: {model_key}")
        return _model_cache[model_key]


class PredictRequest(BaseModel):
    image_base64: str
    model: str = "etiquetas"
    conf: float = 0.25


def decode_image(b64: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")


def validate_etiquetas(detections: list) -> dict:
    """Foto 2 — Etiquetas en FO al ingreso.
    El modelo puede clasificar ambas etiquetas como ETIQUETA 2.
    Aprobado si al menos una etiqueta es visible.
    """
    names = {d["class_name"] for d in detections}
    e1 = "ETIQUETA 1" in names
    e2 = "ETIQUETA 2" in names
    return {
        "etiqueta1_presente": e1,
        "etiqueta2_presente": e2,
        "aprobado": e1 or e2,  # al menos una etiqueta visible
    }


def validate_etiqueta_tapa(detections: list) -> dict:
    """Foto 3 — Etiqueta en tapa exterior de manga."""
    names = {d["class_name"] for d in detections}
    etiqueta_ok = "Etiqueta" in names or "clase_0" in names or len(detections) > 0
    manga_ok    = "Manga"    in names or "clase_1" in names
    return {
        "etiqueta_presente": etiqueta_ok,
        "manga_presente":    manga_ok,
        "aprobado":          etiqueta_ok,
    }


def validate_manga(detections: list) -> dict:
    """Foto 1 — Manga correctamente sellada."""
    names = {d["class_name"] for d in detections}
    manga_ok   = "Manga"     in names
    seguro1_ok = "Seguros 1" in names
    seguro2_ok = "Seguros 2" in names
    tapones_ok = "Tapones"   in names
    aprobado = manga_ok
    return {
        "manga_presente":    manga_ok,
        "seguros_presentes": seguro1_ok or seguro2_ok,
        "seguro1":           seguro1_ok,
        "seguro2":           seguro2_ok,
        "tapones_presentes": tapones_ok,
        "aprobado":          aprobado,
    }


def validate_ubicacion_manga(detections: list) -> dict:
    """Foto 4 — Manga instalada en poste o mensajero."""
    names = {d["class_name"] for d in detections}
    manga_ok     = "MANGA"     in names
    poste_ok     = "POSTE"     in names
    distancia_ok = "DISTANCIA" in names
    return {
        "manga_presente":     manga_ok,
        "poste_presente":     poste_ok,
        "distancia_presente": distancia_ok,
        "aprobado":           manga_ok and poste_ok,
    }


def validate_panoramica_f8(detections: list) -> dict:
    """Foto 5 — Panorámica del poste. Siempre aprobada (foto documental)."""
    names = {d["class_name"] for d in detections}
    manga_ok   = "MANGA"   in names
    reserva_ok = "RESERVA" in names or "1 RESERVA 2" in names
    return {
        "manga_presente":   manga_ok,
        "reserva_presente": reserva_ok,
        "aprobado":         True,
    }


@app.get("/health")
def health():
    base = os.path.dirname(__file__)
    disponibles = [k for k, f in MODEL_FILES.items() if os.path.exists(os.path.join(base, f))]
    cargados    = list(_model_cache.keys())
    return {
        "status":      "ok",
        "disponibles": disponibles,
        "cargados":    cargados,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Recibe imagen en base64 (JSON)."""
    image = decode_image(req.image_base64)
    return _run_model(req.model, image, req.conf)


# Umbrales máximos por modelo — el cliente puede pedir menos, no más
MODEL_MAX_CONF = {
    "etiquetas":      0.25,  # ETIQUETA detectada entre 0.25-0.46 según foto
    "panoramica-f8":  0.05,  # modelo de bajo confidence en vistas aéreas
}

def _run_model(model_key: str, image: Image.Image, conf: float) -> dict:
    if model_key not in MODEL_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"model debe ser uno de: {list(MODEL_FILES.keys())}"
        )
    # Aplicar umbral máximo por modelo (ignorar conf alto enviado por cliente)
    max_conf = MODEL_MAX_CONF.get(model_key)
    if max_conf is not None and conf > max_conf:
        conf = max_conf
    model = _get_model(model_key)
    results = model(image, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class_id":   int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox":       [round(v, 1) for v in box.xyxy[0].tolist()],
            })
    validation = (
        validate_manga(detections)              if model_key == "manga"
        else validate_etiqueta_tapa(detections) if model_key == "etiqueta-tapa"
        else validate_ubicacion_manga(detections) if model_key == "ubicacion-manga"
        else validate_panoramica_f8(detections) if model_key == "panoramica-f8"
        else validate_etiquetas(detections)
    )
    return {
        "model":         model_key,
        "detections":    detections,
        "count":         len(detections),
        "classes_found": list({d["class_name"] for d in detections}),
        "validation":    validation,
    }


# ── Endpoint para N8N: imagen como archivo (multipart/form-data) ──────────────
@app.post("/predict-form")
async def predict_form(
    image: UploadFile = File(...),
    model: str        = Form("etiquetas"),
    conf:  float      = Form(0.25),
):
    try:
        data = await image.read()
        img  = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")
    return _run_model(model, img, conf)


# ── Endpoint para N8N: imagen por URL (JSON) ──────────────────────────────────
class PredictUrlRequest(BaseModel):
    url:          str
    model:        str   = "etiquetas"
    conf:         float = 0.25
    bearer_token: Optional[str] = None

@app.post("/predict-url")
def predict_url(req: PredictUrlRequest):
    headers = {}
    if req.bearer_token:
        headers["Authorization"] = f"Bearer {req.bearer_token}"
    try:
        resp = http_requests.get(req.url, headers=headers, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except http_requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error descargando imagen: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")
    return _run_model(req.model, img, req.conf)
