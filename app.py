from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from typing import Optional, List
import base64, io, os, gc, threading, re, difflib, json
import requests as http_requests

app = FastAPI(title="FiscAI YOLO API — Backup Render — Todos los modelos manga + ODF")

# ── Catálogo de nodos GIS ─────────────────────────────────────────────────────
_NODOS_PATH = os.path.join(os.path.dirname(__file__), 'nodos.json')
try:
    with open(_NODOS_PATH, encoding='utf-8') as _f:
        _nodos_data = json.load(_f)
    NODOS_ABREVIATURAS: List[str] = _nodos_data.get('abreviaturas', [])
    print(f"[nodos] Catálogo cargado: {len(NODOS_ABREVIATURAS)} abreviaturas")
except Exception as _e:
    NODOS_ABREVIATURAS = []
    print(f"[nodos] ADVERTENCIA: no se pudo cargar nodos.json — {_e}")

# ── Lazy loading: 1 modelo en RAM a la vez ────────────────────────────────────
_model_lock  = threading.Lock()
_model_cache: dict = {}

MODEL_FILES = {
    # ── Mangas estándar ──────────────────────────────────────────────────────
    "manga":              "MANGA_SELLADA_V2.pt",
    "etiquetas":          "MANGA_ETIQUETAS_FO_INGRESO_v2.pt",
    "etiqueta-tapa":      "ETIQUETA_TAPA_MANGA_v2.pt",
    "ubicacion-manga":    "UBICACION_MANGA_v2.1.pt",
    "panoramica-f8":      "PANORAMICA_FIGURA_8_V2.pt",
    "casetera-manga":     "CASETERA_MANGA.pt",
    "panoramica-destapada": "PANORAMICA_MANGA_DESTAPADA.pt",
    # ── Mangas 2 hilos ───────────────────────────────────────────────────────
    "manga-2h":           "2H_MANGA_SELLADA.pt",
    "etiqueta-tapa-2h":   "2H_ETIQUETA_TAPA_MANGA.pt",
    "ubicacion-manga-2h": "2H_UBICACION_MANGA.pt",
    "panoramica-f8-2h":   "2H_PANORAMICA_FIGURA_8.pt",
    "ocr-login-2h":       "2H_ETIQUETA_TAPA_MANGA.pt",
    # ── ODF ──────────────────────────────────────────────────────────────────
    "casetera-odf":         "CASETERA_ODF_v1.pt",
    "caseteras-completas":  "CASETERAS_COMPLETAS_v1.pt",
    "odf-aseguramiento":    "INGRESO_FO_AL_ODF_v1.pt",
    "odf-frontal":          "PANORAMICA_FRONTAL_ODF_v1.pt",
    "odf-posterior":        "PANORAMICA_POSTERIOR_ODF_v1.pt",
}

# Mapa Telconet endpoint → model key interno
TN_EP_MAP = {
    "manga_sellada":               "manga",
    "etiquetas_fo_ingreso":        "etiquetas",
    "etiqueta_tapa_manga":         "etiqueta-tapa",
    "ubicacion_manga":             "ubicacion-manga",
    "panoramica_figura_8":         "panoramica-f8",
    "casetera_manga":              "casetera-manga",
    "panoramica_manga_destapada":  "panoramica-destapada",
    "fo_2h_manga_sellada":         "manga-2h",
    "fo_2h_etiqueta_tapa_manga":   "etiqueta-tapa-2h",
    "fo_2h_ubicacion_manga":       "ubicacion-manga-2h",
    "fo_2h_manga_panoramica_figura_8": "panoramica-f8-2h",
    "casetera_odf":                "casetera-odf",
    "caseteras_completas":         "caseteras-completas",
    "ingreso_fo_al_odf":           "odf-aseguramiento",
    "panoramica_frontal_odf":      "odf-frontal",
    "panoramica_posterior_odf":    "odf-posterior",
}

OCR_MODELS = {"etiquetas", "etiqueta-tapa", "etiqueta-tapa-2h", "ocr-login-2h", "odf-frontal"}

LABEL_CLASS = {
    "etiquetas":        "ETIQUETA_FO",
    "etiqueta-tapa":    "ETIQUETA_TAPA",
    "etiqueta-tapa-2h": "etiqueta",
    "ocr-login-2h":     "etiqueta",
    "odf-frontal":      "ETIQUETA_ODF",   # etiqueta GIS en tapa frontal
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


# ── EasyOCR ───────────────────────────────────────────────────────────────────
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


# ── Normalización y fuzzy matching ────────────────────────────────────────────
def normalizar_nomenclatura(texto: str) -> str:
    texto = texto.upper().strip().replace(" ", "")
    texto = texto.replace("=", "-").replace(".", "-")
    texto = re.sub(r'Ø', '0', texto)
    texto = re.sub(r'([A-Z])O(\d)', r'\g<1>0\2', texto)
    texto = re.sub(r'-0([A-Z])', r'-O\1', texto)
    texto = re.sub(r'[^A-Z0-9\-\(\)]', '', texto)
    return texto

def fuzzy_match_gis(texto: str, candidatos: List[str], cutoff: float = 0.80) -> dict:
    if not candidatos or not texto:
        return {"match": None, "score": 0.0, "aprobado": False}
    texto_norm      = normalizar_nomenclatura(texto)
    candidatos_norm = [normalizar_nomenclatura(c) for c in candidatos]
    matches = difflib.get_close_matches(texto_norm, candidatos_norm, n=1, cutoff=cutoff)
    if matches:
        idx   = candidatos_norm.index(matches[0])
        score = difflib.SequenceMatcher(None, texto_norm, matches[0]).ratio()
        return {"match": candidatos[idx], "match_normalizado": matches[0],
                "texto_normalizado": texto_norm, "score": round(score, 3), "aprobado": True}
    best_norm = max(candidatos_norm, key=lambda c: difflib.SequenceMatcher(None, texto_norm, c).ratio())
    score = difflib.SequenceMatcher(None, texto_norm, best_norm).ratio()
    idx   = candidatos_norm.index(best_norm)
    return {"match": candidatos[idx], "match_normalizado": best_norm,
            "texto_normalizado": texto_norm, "score": round(score, 3), "aprobado": False}


# ── OCR sobre bboxes ──────────────────────────────────────────────────────────
def _run_ocr(model_key: str, image: Image.Image,
             detections: list, gis_nombres: List[str]) -> list:
    label_class = LABEL_CLASS.get(model_key, "ETIQUETA_FO")
    etiquetas   = [d for d in detections if d["class_name"] == label_class]
    if not etiquetas:
        return []
    if model_key == "etiquetas" and NODOS_ABREVIATURAS:
        candidatos = NODOS_ABREVIATURAS
    elif gis_nombres:
        candidatos = gis_nombres
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
        entry = {"etiqueta_idx": i, "bbox": det["bbox"],
                 "confianza_yolo": det["confidence"], "texto_raw": texto_raw,
                 "texto_normalizado": normalizar_nomenclatura(texto_raw) if texto_raw else ""}
        if candidatos and texto_raw:
            entry["gis_match"] = fuzzy_match_gis(texto_raw, candidatos)
        results.append(entry)
    return results


# ── Validadores ───────────────────────────────────────────────────────────────

def validate_manga(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    return {"manga_presente": "Manga" in names,
            "seguros_presentes": "Seguros 1" in names or "Seguros 2" in names,
            "seguro1": "Seguros 1" in names, "seguro2": "Seguros 2" in names,
            "tapones_presentes": "Tapones" in names,
            "aprobado": "Manga" in names}

def validate_etiquetas(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    return {"etiqueta_fo_presente": "ETIQUETA_FO" in names, "manga_presente": "MANGA" in names,
            "cantidad_etiquetas": sum(1 for d in detections if d["class_name"] == "ETIQUETA_FO"),
            "aprobado": "ETIQUETA_FO" in names}

def validate_etiqueta_tapa(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    return {"etiqueta_tapa_presente": "ETIQUETA_TAPA" in names,
            "manga_presente": "MANGA" in names, "aprobado": "ETIQUETA_TAPA" in names}

def validate_ubicacion_manga(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    return {"manga_presente": "MANGA" in names, "poste_presente": "POSTE" in names,
            "aprobado": "MANGA" in names and "POSTE" in names}

def validate_panoramica_f8(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    return {"figura8_presente": "FIGURA_8" in names, "manga_presente": "MANGA" in names,
            "aprobado": True}

def validate_casetera_manga(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    return {"casetera_presente": bool(detections),
            "clases": list(names),
            "aprobado": bool(detections)}

def validate_panoramica_destapada(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    no_conforme = "NO_CONFORME" in names or "PARCIALMENTE_CONFORME" in names
    return {"clases": list(names), "no_conforme": no_conforme,
            "detecciones": len(detections), "aprobado": bool(detections) and not no_conforme}

# ── Validadores 2H ────────────────────────────────────────────────────────────
def validate_manga_2h(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    manga_ok    = "manga_completa"     in names
    sellado_ok  = "sellado_correcto"   in names
    sin_sellado = "sellado_incorrecto" in names
    return {"manga_presente": manga_ok, "sellado_correcto": sellado_ok,
            "sellado_incorrecto": sin_sellado, "amarras_presentes": "amarra" in names,
            "aprobado": manga_ok and sellado_ok and not sin_sellado}

def validate_etiqueta_tapa_2h(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    return {"etiqueta_presente": "etiqueta" in names or bool(detections),
            "manga_presente": "manga_2hilos" in names,
            "aprobado": "etiqueta" in names or bool(detections)}

def validate_ubicacion_manga_2h(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    ub_incorrecta = "ubicacion_incorrecta" in names
    return {"manga_presente": "manga" in names, "poste_presente": "poste" in names,
            "ubicacion_correcta": "ubicacion_correcta" in names,
            "ubicacion_incorrecta": ub_incorrecta,
            "aprobado": "manga" in names and ("poste" in names or "ubicacion_correcta" in names) and not ub_incorrecta}

def validate_panoramica_f8_2h(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    figura8_wrong = "figura8_incorrecta" in names
    return {"manga_presente": "manga" in names, "figura8_presente": "figura8" in names,
            "figura8_incorrecta": figura8_wrong,
            "aprobado": ("figura8" in names or "manga" in names) and not figura8_wrong}

# ── Validadores ODF ───────────────────────────────────────────────────────────
def validate_odf_generico(detections: list) -> dict:
    """Validador genérico para modelos ODF: aprobado si hay ≥1 detección."""
    names = {d["class_name"] for d in detections}
    return {"clases": list(names), "detecciones": len(detections),
            "aprobado": bool(detections)}

def validate_caseteras_completas(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    return {"clases": list(names), "detecciones": len(detections),
            "cantidad_caseteras": sum(1 for d in detections if "casetera" in d["class_name"].lower()),
            "aprobado": bool(detections)}

def validate_odf_frontal(detections: list) -> dict:
    names = {d["class_name"] for d in detections}
    return {"etiqueta_odf_presente": "ETIQUETA_ODF" in names,
            "clases": list(names), "detecciones": len(detections),
            "aprobado": bool(detections)}

# ── Umbrales máximos por modelo ───────────────────────────────────────────────
MODEL_MAX_CONF = {
    "panoramica-f8":    0.05,
    "panoramica-f8-2h": 0.05,
}

VALIDATORS = {
    "manga":              validate_manga,
    "etiquetas":          validate_etiquetas,
    "etiqueta-tapa":      validate_etiqueta_tapa,
    "ubicacion-manga":    validate_ubicacion_manga,
    "panoramica-f8":      validate_panoramica_f8,
    "casetera-manga":     validate_casetera_manga,
    "panoramica-destapada": validate_panoramica_destapada,
    "manga-2h":           validate_manga_2h,
    "etiqueta-tapa-2h":   validate_etiqueta_tapa_2h,
    "ubicacion-manga-2h": validate_ubicacion_manga_2h,
    "panoramica-f8-2h":   validate_panoramica_f8_2h,
    "ocr-login-2h":       validate_etiqueta_tapa_2h,
    "casetera-odf":        validate_odf_generico,
    "caseteras-completas": validate_caseteras_completas,
    "odf-aseguramiento":   validate_odf_generico,
    "odf-frontal":         validate_odf_frontal,
    "odf-posterior":       validate_odf_generico,
}

def decode_image(b64: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")

def _run_model(model_key: str, image: Image.Image,
               conf: float, gis_nombres: List[str] = []) -> dict:
    if model_key not in MODEL_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"model debe ser uno de: {list(MODEL_FILES.keys())}"
        )
    max_conf = MODEL_MAX_CONF.get(model_key)
    if max_conf is not None and conf > max_conf:
        conf = max_conf

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

    validator  = VALIDATORS.get(model_key, validate_odf_generico)
    validation = validator(detections)

    response = {
        "model":         model_key,
        "detections":    detections,
        "count":         len(detections),
        "classes_found": list({d["class_name"] for d in detections}),
        "validation":    validation,
    }

    if model_key in OCR_MODELS:
        ocr = _run_ocr(model_key, image, detections, gis_nombres)
        response["ocr"] = ocr
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
        "status":      "ok",
        "disponibles": disponibles,
        "cargados":    list(_model_cache.keys()),
        "ocr_listo":   _ocr_reader is not None,
        "nodos_gis":   len(NODOS_ABREVIATURAS),
    }


# ── Endpoint compatible con Telconet (mismo formato URL + respuesta) ───────────
# Permite usar este servicio como drop-in replacement de la API Telconet:
#   POST /predict/fibra/{endpoint}/json?confidence={conf}
# Body: multipart/form-data, campo "file" (imagen)
# Respuesta: { detections_count, detections: [{class_name, confidence, bbox}] }
@app.post("/predict/fibra/{tn_endpoint}/json")
async def predict_telconet_compat(
    tn_endpoint: str,
    confidence:  float      = 0.25,
    file:        UploadFile = File(...),
):
    model_key = TN_EP_MAP.get(tn_endpoint)
    if not model_key:
        raise HTTPException(
            status_code=404,
            detail=f"Endpoint Telconet desconocido: '{tn_endpoint}'. Disponibles: {list(TN_EP_MAP.keys())}"
        )
    try:
        data = await file.read()
        img  = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")

    r = _run_model(model_key, img, confidence)
    # Respuesta con el MISMO formato que Telconet para que yolo-client.js no necesite cambios.
    return {
        "detections_count": r["count"],
        "detections":       r["detections"],
        "_source":          "render-backup",
        "_model_key":       model_key,
    }


@app.post("/predict")
def predict(req: "PredictRequest"):
    image = decode_image(req.image_base64)
    return _run_model(req.model, image, req.conf, req.gis_nombres)


@app.post("/predict-form")
async def predict_form(
    image:       UploadFile = File(...),
    model:       str        = Form("etiquetas"),
    conf:        float      = Form(0.25),
    gis_nombres: str        = Form(""),
):
    try:
        data = await image.read()
        img  = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")
    nombres = json.loads(gis_nombres) if gis_nombres.strip() else []
    return _run_model(model, img, conf, nombres)


class PredictUrlRequest(BaseModel):
    url:          str
    model:        str       = "etiquetas"
    conf:         float     = 0.25
    bearer_token: Optional[str] = None
    gis_nombres:  List[str] = []

class PredictRequest(BaseModel):
    image_base64: str
    model:        str       = "etiquetas"
    conf:         float     = 0.25
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
