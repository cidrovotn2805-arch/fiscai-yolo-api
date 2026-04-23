from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image
import base64, io, os

app = FastAPI(title="FiscAI YOLO API — Etiquetas FO + Manga Detector")

# ── Cargar ambos modelos al iniciar ──────────────────────────────────────────
def load(repo: str, filename: str = "best.pt") -> YOLO:
    path = hf_hub_download(repo_id=repo, filename=filename)
    return YOLO(path)

print("Cargando modelos...")
MODEL_ETIQUETAS    = load("cidrovo/etiquetas-fo-ingreso")   # ETIQUETA 1, ETIQUETA 2, MANGA
MODEL_MANGA        = load("cidrovo/manga-detector")          # Manga, Seguros 1, Seguros 2, Tapones
MODEL_ETIQUETA_TAPA = load("cidrovo/ETIQUETA_TAPA_MANGA")   # Etiqueta, Manga
print("Modelos listos.")

MODELS = {
    "etiquetas":    MODEL_ETIQUETAS,
    "manga":        MODEL_MANGA,
    "etiqueta-tapa": MODEL_ETIQUETA_TAPA,
}


class PredictRequest(BaseModel):
    image_base64: str        # base64 sin prefijo data:...
    model: str = "etiquetas" # "etiquetas" | "manga"
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
    # Acepta nombres entrenados O genéricos (clase_0, clase_1)
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
    aprobado   = manga_ok and (seguro1_ok or seguro2_ok) and tapones_ok
    return {
        "manga_presente":   manga_ok,
        "seguros_presentes": seguro1_ok or seguro2_ok,
        "seguro1":          seguro1_ok,
        "seguro2":          seguro2_ok,
        "tapones_presentes": tapones_ok,
        "aprobado":         aprobado,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {k: list(m.names.values()) for k, m in MODELS.items()}
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if req.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"model debe ser: {list(MODELS.keys())}")

    model  = MODELS[req.model]
    image  = decode_image(req.image_base64)
    results = model(image, conf=req.conf, verbose=False)

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
        validate_manga(detections)         if req.model == "manga"
        else validate_etiqueta_tapa(detections) if req.model == "etiqueta-tapa"
        else validate_etiquetas(detections)
    )

    return {
        "model":         req.model,
        "detections":    detections,
        "count":         len(detections),
        "classes_found": list({d["class_name"] for d in detections}),
        "validation":    validation,
    }
