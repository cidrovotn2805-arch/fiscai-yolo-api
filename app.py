from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from typing import Optional
import base64, io, os, requests as http_requests

app = FastAPI(title="FiscAI YOLO API — Todos los modelos de manga")

# ── Carga segura desde archivo local ─────────────────────────────────────────
def load_local(filename: str) -> Optional[YOLO]:
    try:
        return YOLO(os.path.join(os.path.dirname(__file__), filename))
    except Exception as e:
        print(f"⚠  No se pudo cargar {filename}: {e}")
        return None

print("Cargando modelos...")
MODEL_MANGA         = load_local("MANGA_SELLADA.pt")          # Foto 1: Manga, Seguros 1, Seguros 2, Tapones
MODEL_ETIQUETAS     = load_local("ETIQUETAS_FO_INGRESO.pt")   # Foto 2: ETIQUETA 1, ETIQUETA 2, MANGA
MODEL_ETIQUETA_TAPA = load_local("ETIQUETA_TAPA_MANGA.pt")    # Foto 3: Etiqueta, Manga
MODEL_UBICACION     = load_local("UBICACION_MANGA.pt")        # Foto 4: DISTANCIA, MANGA, POSTE
MODEL_PANORAMICA    = load_local("PANORAMICA_FIGURA_8.pt")    # Foto 5: MANGA, RESERVA, 1 RESERVA 2
print("Modelos listos.")

MODELS = {
    "etiquetas":       MODEL_ETIQUETAS,
    "manga":           MODEL_MANGA,
    "etiqueta-tapa":   MODEL_ETIQUETA_TAPA,
    "ubicacion-manga": MODEL_UBICACION,
    "panoramica-f8":   MODEL_PANORAMICA,
}


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
    """Foto 2 — Etiquetas en FO al ingreso."""
    names = {d["class_name"] for d in detections}
    e1 = "ETIQUETA 1" in names
    e2 = "ETIQUETA 2" in names
    return {
        "etiqueta1_presente": e1,
        "etiqueta2_presente": e2,
        "aprobado": e1 and e2,
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
    """Foto 1 — Manga correctamente sellada.
    Aprueba si la manga es visible en la foto.
    """
    names = {d["class_name"] for d in detections}
    manga_ok   = "Manga"     in names
    seguro1_ok = "Seguros 1" in names
    seguro2_ok = "Seguros 2" in names
    tapones_ok = "Tapones"   in names
    # Aprueba con solo detectar la manga — criterio principal de foto 1
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
    """Foto 4 — Manga instalada en poste o mensajero.
    Clases del modelo: DISTANCIA, MANGA, POSTE.
    Aprueba si la manga Y el poste son visibles.
    """
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
    """Foto 5 — Panorámica del poste. Siempre aprobada (foto documental).
    Clases del modelo: MANGA, RESERVA, 1 RESERVA 2.
    """
    names = {d["class_name"] for d in detections}
    manga_ok   = "MANGA"   in names
    reserva_ok = "RESERVA" in names or "1 RESERVA 2" in names
    return {
        "manga_presente":   manga_ok,
        "reserva_presente": reserva_ok,
        "aprobado":         True,   # Foto documental — siempre aprobada
    }


@app.get("/health")
def health():
    disponibles = {k: list(m.names.values()) for k, m in MODELS.items() if m is not None}
    faltantes   = [k for k, m in MODELS.items() if m is None]
    return {
        "status":      "ok" if not faltantes else "degradado",
        "disponibles": disponibles,
        "faltantes":   faltantes,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Recibe imagen en base64 (JSON)."""
    image = decode_image(req.image_base64)
    return _run_model(req.model, image, req.conf)


def _run_model(model_key: str, image: Image.Image, conf: float) -> dict:
    """Lógica común de inferencia."""
    if model_key not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model debe ser uno de: {list(MODELS.keys())}"
        )
    model = MODELS[model_key]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo '{model_key}' no disponible"
        )
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
