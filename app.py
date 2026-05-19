from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from typing import Optional, List
import base64, io, os, gc, threading, re, difflib, json
import requests as http_requests

app = FastAPI(title="FiscAI YOLO API — Todos los modelos de manga")

# ── Catálogo de nodos GIS (GYE + TS + UIO) ───────────────────────────────────
_NODOS_PATH = os.path.join(os.path.dirname(__file__), 'nodos.json')
try:
    with open(_NODOS_PATH, encoding='utf-8') as _f:
        _nodos_data = json.load(_f)
    NODOS_ABREVIATURAS: List[str] = _nodos_data.get('abreviaturas', [])
    print(f"[nodos] Catálogo cargado: {len(NODOS_ABREVIATURAS)} abreviaturas")
except Exception as _e:
    NODOS_ABREVIATURAS = []
    print(f"[nodos] ADVERTENCIA: no se pudo cargar nodos.json — {_e}")

# ── Lazy loading YOLO: solo 1 modelo en memoria a la vez ─────────────────────
_model_lock  = threading.Lock()
_model_cache: dict = {}

MODEL_FILES = {
    # ── Mangas estándar (144H / 48H / 24H) ──────────────────────────────────
    "manga":           "MANGA_SELLADA.pt",
    "etiquetas":       "ETIQUETAS_FO_INGRESO.pt",
    "etiqueta-tapa":   "ETIQUETA_TAPA_MANGA.pt",
    "ubicacion-manga": "UBICACION_MANGA.pt",
    "panoramica-f8":   "PANORAMICA_FIGURA_8.pt",
    # ── Mangas de 2 hilos ───────────────────────────────────────────────────
    "manga-2h":           "2H_MANGA_SELLADA.pt",
    "etiqueta-tapa-2h":   "2H_ETIQUETA_TAPA_MANGA.pt",
    "ubicacion-manga-2h": "2H_UBICACION_MANGA.pt",
    "panoramica-f8-2h":   "2H_PANORAMICA_FIGURA_8.pt",
}

# Modelos que requieren lectura de etiqueta por OCR
OCR_MODELS = {"etiquetas", "etiqueta-tapa", "etiqueta-tapa-2h"}

# Clase de etiqueta según modelo
LABEL_CLASS = {
    "etiquetas":        "ETIQUETA_FO",
    "etiqueta-tapa":    "ETIQUETA_TAPA",
    "etiqueta-tapa-2h": "etiqueta",
}

def _get_model(model_key: str) -> YOLO:
    with _model_lock:
        if model_key not in _model_cache:
            for k in list(_model_cache.keys()):
                del _model_cache[k]
            gc.collect()
            path = os.path.join(os.path.dirname(__file__), MODEL_FILES[model_key])
            print(f"[lazy] Cargando modelo: {model_key} ({MODEL_FILES[model_key]})")
            _model_cache[model_key] = YOLO(path)
            print(f"[lazy] Modelo listo: {model_key}")
        return _model_cache[model_key]


# ── EasyOCR (lazy load) ───────────────────────────────────────────────────────
_ocr_lock   = threading.Lock()
_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    with _ocr_lock:
        if _ocr_reader is None:
            import easyocr
            print("[OCR] Cargando EasyOCR reader (es+en)...")
            _ocr_reader = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)
            print("[OCR] EasyOCR listo.")
    return _ocr_reader


# ── Normalización y fuzzy matching ───────────────────────────────────────────

def normalizar_nomenclatura(texto: str) -> str:
    """Limpia y normaliza el texto OCR para comparar contra GIS."""
    texto = texto.upper().strip()
    texto = texto.replace(" ", "")
    texto = texto.replace("=", "-").replace(".", "-")
    # Ø y O → 0 cuando están entre letras y dígitos (ej: WMP-Ø2 → WMP-02)
    texto = re.sub(r'Ø', '0', texto)
    texto = re.sub(r'([A-Z])O(\d)', r'\g<1>0\2', texto)
    # 0 → O cuando está después de guión y antes de letras (ej: -0A → -OA)
    texto = re.sub(r'-0([A-Z])', r'-O\1', texto)
    # Solo caracteres válidos en nomenclatura
    texto = re.sub(r'[^A-Z0-9\-\(\)]', '', texto)
    return texto

def fuzzy_match_gis(texto: str, candidatos: List[str], cutoff: float = 0.80) -> dict:
    """Busca el mejor match fuzzy de texto_normalizado en la lista GIS."""
    if not candidatos or not texto:
        return {"match": None, "score": 0.0, "aprobado": False}

    texto_norm       = normalizar_nomenclatura(texto)
    candidatos_norm  = [normalizar_nomenclatura(c) for c in candidatos]

    matches = difflib.get_close_matches(texto_norm, candidatos_norm, n=1, cutoff=cutoff)
    if matches:
        idx   = candidatos_norm.index(matches[0])
        score = difflib.SequenceMatcher(None, texto_norm, matches[0]).ratio()
        return {
            "match":             candidatos[idx],
            "match_normalizado": matches[0],
            "texto_normalizado": texto_norm,
            "score":             round(score, 3),
            "aprobado":          True,
        }

    # Sin match aceptable — igual retorna el mejor para diagnóstico
    best_norm = max(candidatos_norm,
                    key=lambda c: difflib.SequenceMatcher(None, texto_norm, c).ratio())
    score = difflib.SequenceMatcher(None, texto_norm, best_norm).ratio()
    idx   = candidatos_norm.index(best_norm)
    return {
        "match":             candidatos[idx],
        "match_normalizado": best_norm,
        "texto_normalizado": texto_norm,
        "score":             round(score, 3),
        "aprobado":          False,
    }


# ── OCR sobre bboxes de etiquetas detectadas ─────────────────────────────────

def _run_ocr(model_key: str, image: Image.Image,
             detections: list, gis_nombres: List[str]) -> list:
    """Recorta cada etiqueta detectada, corre OCR y hace fuzzy match.

    - etiquetas (foto 2): valida contra catálogo completo de nodos GIS (1 100+ abrev.)
    - etiqueta-tapa (foto 3): valida contra gis_nombres (nomenclatura del técnico)
    """
    label_class = LABEL_CLASS.get(model_key, "ETIQUETA_FO")
    etiquetas   = [d for d in detections if d["class_name"] == label_class]

    if not etiquetas:
        return []

    # Determinar lista de candidatos según el modelo
    if model_key == "etiquetas" and NODOS_ABREVIATURAS:
        candidatos = NODOS_ABREVIATURAS          # catálogo completo ~1 100 abreviaturas
    elif gis_nombres:
        candidatos = gis_nombres                 # nomenclatura ingresada por el técnico
    else:
        candidatos = []

    reader  = _get_ocr_reader()
    w, h    = image.size
    results = []

    for i, det in enumerate(etiquetas):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        x1 = max(0, x1 - 10);  y1 = max(0, y1 - 10)
        x2 = min(w, x2 + 10);  y2 = min(h, y2 + 10)

        try:
            crop      = image.crop((x1, y1, x2, y2))
            textos    = reader.readtext(crop, detail=0, paragraph=True)
            texto_raw = " ".join(textos).strip()
        except Exception as e:
            print(f"[OCR] Error en etiqueta {i}: {e}")
            texto_raw = ""

        entry = {
            "etiqueta_idx":      i,
            "bbox":              det["bbox"],
            "confianza_yolo":    det["confidence"],
            "texto_raw":         texto_raw,
            "texto_normalizado": normalizar_nomenclatura(texto_raw) if texto_raw else "",
        }

        if candidatos and texto_raw:
            entry["gis_match"] = fuzzy_match_gis(texto_raw, candidatos)

        results.append(entry)

    return results


# ── Request / Response ────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    image_base64: str
    model:        str        = "etiquetas"
    conf:         float      = 0.25
    gis_nombres:  List[str]  = []   # nomenclaturas válidas del GIS para validar OCR


def decode_image(b64: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")


# ── Funciones de validación YOLO ─────────────────────────────────────────────

def validate_manga(detections: list) -> dict:
    """Foto 1 — Manga correctamente sellada."""
    names = {d["class_name"] for d in detections}
    manga_ok   = "Manga"     in names
    seguro1_ok = "Seguros 1" in names
    seguro2_ok = "Seguros 2" in names
    tapones_ok = "Tapones"   in names
    return {
        "manga_presente":    manga_ok,
        "seguros_presentes": seguro1_ok or seguro2_ok,
        "seguro1":           seguro1_ok,
        "seguro2":           seguro2_ok,
        "tapones_presentes": tapones_ok,
        "aprobado":          manga_ok,
    }


def validate_etiquetas(detections: list) -> dict:
    """Foto 2 — Etiquetas en FO al ingreso.
    Modelo v2: clases ETIQUETA_FO(0) y MANGA(1).
    """
    names = {d["class_name"] for d in detections}
    etiqueta_ok      = "ETIQUETA_FO" in names
    manga_ok         = "MANGA"       in names
    count_etiquetas  = sum(1 for d in detections if d["class_name"] == "ETIQUETA_FO")
    return {
        "etiqueta_fo_presente": etiqueta_ok,
        "manga_presente":       manga_ok,
        "cantidad_etiquetas":   count_etiquetas,
        "aprobado":             etiqueta_ok,
    }


def validate_etiqueta_tapa(detections: list) -> dict:
    """Foto 3 — Etiqueta en tapa exterior de manga.
    Modelo v2: clases ETIQUETA_TAPA(0) y MANGA(1).
    """
    names = {d["class_name"] for d in detections}
    etiqueta_ok = "ETIQUETA_TAPA" in names
    manga_ok    = "MANGA"         in names
    return {
        "etiqueta_tapa_presente": etiqueta_ok,
        "manga_presente":         manga_ok,
        "aprobado":               etiqueta_ok,
    }


def validate_ubicacion_manga(detections: list) -> dict:
    """Foto 4 — Manga instalada en poste o mensajero."""
    names = {d["class_name"] for d in detections}
    manga_ok = "MANGA" in names
    poste_ok = "POSTE" in names
    return {
        "manga_presente": manga_ok,
        "poste_presente": poste_ok,
        "aprobado":       manga_ok and poste_ok,
    }


def validate_panoramica_f8(detections: list) -> dict:
    """Foto 5 — Panorámica figura 8. Rechaza si no detecta FIGURA_8.
    Modelo v2: clases FIGURA_8(0) y MANGA(1).
    """
    names = {d["class_name"] for d in detections}
    figura8_ok = "FIGURA_8" in names
    return {
        "figura8_presente": figura8_ok,
        "manga_presente":   "MANGA" in names,
        "aprobado":         figura8_ok,
    }


# ── Validadores 2H ───────────────────────────────────────────────────────────

def validate_manga_2h(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    manga_ok    = "manga_completa"    in names
    sellado_ok  = "sellado_correcto"  in names
    sin_sellado = "sellado_incorrecto" in names
    amarra_ok   = "amarra"            in names
    return {
        "manga_presente":     manga_ok,
        "sellado_correcto":   sellado_ok,
        "sellado_incorrecto": sin_sellado,
        "amarras_presentes":  amarra_ok,
        "aprobado":           manga_ok and sellado_ok and not sin_sellado,
    }


def validate_etiqueta_tapa_2h(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    etiqueta_ok = "etiqueta" in names
    manga_ok    = "manga_2hilos" in names
    return {
        "etiqueta_presente": etiqueta_ok,
        "manga_presente":    manga_ok,
        "aprobado":          etiqueta_ok,
    }


def validate_ubicacion_manga_2h(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    manga_ok      = "manga"               in names
    poste_ok      = "poste"               in names
    ub_correcta   = "ubicacion_correcta"  in names
    ub_incorrecta = "ubicacion_incorrecta" in names
    return {
        "manga_presente":       manga_ok,
        "poste_presente":       poste_ok,
        "ubicacion_correcta":   ub_correcta,
        "ubicacion_incorrecta": ub_incorrecta,
        "aprobado":             manga_ok and (poste_ok or ub_correcta) and not ub_incorrecta,
    }


def validate_panoramica_f8_2h(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    figura8_ok    = "figura8" in names
    figura8_wrong = "figura8_incorrecta" in names
    manga_ok      = "manga"  in names
    return {
        "manga_presente":     manga_ok,
        "figura8_presente":   figura8_ok,
        "figura8_incorrecta": figura8_wrong,
        "aprobado":           figura8_ok and not figura8_wrong,
    }


# ── Motor de inferencia ───────────────────────────────────────────────────────

MODEL_MAX_CONF = {}

VALIDATORS = {
    "manga":              validate_manga,
    "manga-2h":           validate_manga_2h,
    "etiquetas":          validate_etiquetas,
    "etiqueta-tapa":      validate_etiqueta_tapa,
    "etiqueta-tapa-2h":   validate_etiqueta_tapa_2h,
    "ubicacion-manga":    validate_ubicacion_manga,
    "ubicacion-manga-2h": validate_ubicacion_manga_2h,
    "panoramica-f8":      validate_panoramica_f8,
    "panoramica-f8-2h":   validate_panoramica_f8_2h,
}

def _run_model(model_key: str, image: Image.Image,
               conf: float, gis_nombres: List[str] = []) -> dict:
    if model_key not in MODEL_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"model debe ser uno de: {list(MODEL_FILES.keys())}"
        )
    # Aplicar umbral máximo por modelo
    max_conf = MODEL_MAX_CONF.get(model_key)
    if max_conf is not None and conf > max_conf:
        conf = max_conf

    # ── YOLO ──────────────────────────────────────────────────────────────────
    model  = _get_model(model_key)
    result = model(image, conf=conf, verbose=False)
    detections = []
    for r in result:
        for box in r.boxes:
            detections.append({
                "class_id":   int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox":       [round(v, 1) for v in box.xyxy[0].tolist()],
            })

    validator  = VALIDATORS.get(model_key, validate_etiquetas)
    validation = validator(detections)

    response = {
        "model":         model_key,
        "detections":    detections,
        "count":         len(detections),
        "classes_found": list({d["class_name"] for d in detections}),
        "validation":    validation,
    }

    # ── OCR (solo para modelos de etiquetas) ──────────────────────────────────
    if model_key in OCR_MODELS:
        ocr = _run_ocr(model_key, image, detections, gis_nombres)
        response["ocr"] = ocr
        # Si hay OCR y GIS, enriquecer la validación con resultado OCR
        if ocr and gis_nombres:
            ocr_aprobados = [e["gis_match"]["aprobado"] for e in ocr if "gis_match" in e]
            response["validation"]["ocr_nomenclatura_ok"] = any(ocr_aprobados)
            response["validation"]["aprobado"] = (
                response["validation"]["aprobado"] and any(ocr_aprobados)
            )

    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    base = os.path.dirname(__file__)
    disponibles = [k for k, f in MODEL_FILES.items() if os.path.exists(os.path.join(base, f))]
    return {
        "status":        "ok",
        "disponibles":   disponibles,
        "cargados":      list(_model_cache.keys()),
        "ocr_listo":     _ocr_reader is not None,
        "nodos_gis":     len(NODOS_ABREVIATURAS),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Recibe imagen en base64 + lista opcional de nomenclaturas GIS."""
    image = decode_image(req.image_base64)
    return _run_model(req.model, image, req.conf, req.gis_nombres)


@app.post("/predict-form")
async def predict_form(
    image:       UploadFile  = File(...),
    model:       str         = Form("etiquetas"),
    conf:        float       = Form(0.25),
    gis_nombres: str         = Form(""),   # JSON array como string: '["WMP-01","WMP-02"]'
):
    try:
        data = await image.read()
        img  = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")
    import json
    nombres = json.loads(gis_nombres) if gis_nombres.strip() else []
    return _run_model(model, img, conf, nombres)


class PredictUrlRequest(BaseModel):
    url:          str
    model:        str       = "etiquetas"
    conf:         float     = 0.25
    bearer_token: Optional[str] = None
    gis_nombres:  List[str] = []

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
    return _run_model(req.model, img, req.conf, req.gis_nombres)
