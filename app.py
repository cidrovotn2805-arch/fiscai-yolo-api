from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image
from typing import Optional, List
import base64, io, os, gc, threading, re, difflib, json, ast
import numpy as np
import onnxruntime as ort
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
_model_cache: dict = {}  # key → (session, class_names, input_name, imgsz)

MODEL_FILES = {
    # ── Mangas estándar ──────────────────────────────────────────────────────
    "manga":              "MANGA_SELLADA_V2.onnx",
    "etiquetas":          "MANGA_ETIQUETAS_FO_INGRESO_v2.onnx",
    "etiqueta-tapa":      "ETIQUETA_TAPA_MANGA_v2.onnx",
    "ubicacion-manga":    "UBICACION_MANGA_v2.1.onnx",
    "panoramica-f8":      "PANORAMICA_FIGURA_8_V2.onnx",
    "casetera-manga":     "CASETERA_MANGA.onnx",
    "panoramica-destapada": "PANORAMICA_MANGA_DESTAPADA.onnx",
    # ── Mangas 2 hilos ───────────────────────────────────────────────────────
    "manga-2h":           "2H_MANGA_SELLADA.onnx",
    "etiqueta-tapa-2h":   "2H_ETIQUETA_TAPA_MANGA.onnx",
    "ubicacion-manga-2h": "2H_UBICACION_MANGA.onnx",
    "panoramica-f8-2h":   "2H_PANORAMICA_FIGURA_8.onnx",
    "ocr-login-2h":       "2H_ETIQUETA_TAPA_MANGA.onnx",
    # ── ODF ──────────────────────────────────────────────────────────────────
    "casetera-odf":         "CASETERA_ODF_v1.onnx",
    "caseteras-completas":  "CASETERAS_COMPLETAS_v1.onnx",
    "odf-aseguramiento":    "INGRESO_FO_AL_ODF_v1.onnx",
    "odf-frontal":          "PANORAMICA_FRONTAL_ODF_v1.onnx",
    "odf-posterior":        "PANORAMICA_POSTERIOR_ODF_v1.onnx",
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
    "odf-frontal":      "ETIQUETA_ODF",
}


# ── Inferencia ONNX pura (sin torch/ultralytics) ──────────────────────────────

def _get_model(model_key: str):
    """Retorna (session, class_names, input_name, imgsz). 1 modelo en RAM."""
    with _model_lock:
        if model_key not in _model_cache:
            for k in list(_model_cache.keys()):
                del _model_cache[k]
            gc.collect()
            path = os.path.join(os.path.dirname(__file__), MODEL_FILES[model_key])
            print(f"[lazy] Cargando ONNX: {model_key} ({MODEL_FILES[model_key]})", flush=True)
            session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])

            # Nombres de clases embebidos en el metadata por ultralytics
            meta = session.get_modelmeta().custom_metadata_map
            names_str = meta.get('names', '{}')
            try:
                names_dict = ast.literal_eval(names_str)
                class_names = [names_dict[i] for i in sorted(names_dict.keys())]
            except Exception:
                class_names = []

            # Tamaño de entrada: shape [1, 3, H, W]
            inp = session.get_inputs()[0]
            input_name = inp.name
            try:
                imgsz = int(inp.shape[2])
                if imgsz <= 0:
                    imgsz = 640
            except (TypeError, ValueError, IndexError):
                imgsz = 640

            print(f"[lazy] Listo: {model_key} imgsz={imgsz} clases={class_names}", flush=True)
            _model_cache[model_key] = (session, class_names, input_name, imgsz)
        return _model_cache[model_key]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        inter_x1 = np.maximum(x1[i], x1[order[1:]])
        inter_y1 = np.maximum(y1[i], y1[order[1:]])
        inter_x2 = np.minimum(x2[i], x2[order[1:]])
        inter_y2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_threshold]
    return keep


def _run_inference(model_key: str, image: Image.Image, conf: float) -> list:
    """Inferencia ONNX. YOLOv8 output: [1, 4+nc, 8400]."""
    session, class_names, input_name, imgsz = _get_model(model_key)

    orig_w, orig_h = image.size
    img_r = image.resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.array(img_r, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)       # HWC → CHW
    arr = np.expand_dims(arr, axis=0)  # [1, 3, H, W]

    outputs = session.run(None, {input_name: arr})
    out = outputs[0][0].T  # [8400, 4+nc]

    xywh        = out[:, :4]
    class_scores = out[:, 4:]
    class_ids   = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences >= conf
    if not mask.any():
        return []
    xywh        = xywh[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    cx, cy, bw, bh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = (cx - bw / 2) / imgsz * orig_w
    y1 = (cy - bh / 2) / imgsz * orig_h
    x2 = (cx + bw / 2) / imgsz * orig_w
    y2 = (cy + bh / 2) / imgsz * orig_h
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    detections = []
    for cls_id in np.unique(class_ids):
        cmask = class_ids == cls_id
        cb, cs = boxes[cmask], confidences[cmask]
        for idx in _nms(cb, cs):
            name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(cls_id)
            detections.append({
                "class_id":   int(cls_id),
                "class_name": name,
                "confidence": round(float(cs[idx]), 4),
                "bbox":       [round(float(cb[idx, 0]), 1), round(float(cb[idx, 1]), 1),
                               round(float(cb[idx, 2]), 1), round(float(cb[idx, 3]), 1)],
            })
    return detections


# ── EasyOCR (opcional — solo si está instalado con torch) ─────────────────────
_ocr_lock    = threading.Lock()
_ocr_reader  = None
_ocr_disabled = False

def _get_ocr_reader():
    global _ocr_reader, _ocr_disabled
    if _ocr_disabled:
        return None
    with _ocr_lock:
        if _ocr_reader is None and not _ocr_disabled:
            try:
                import easyocr
                print("[OCR] Cargando EasyOCR reader (es+en)...")
                _ocr_reader = easyocr.Reader(['es', 'en'], gpu=False, verbose=False)
                print("[OCR] EasyOCR listo.")
            except Exception as e:
                print(f"[OCR] No disponible ({e}) — deshabilitado en backup.")
                _ocr_disabled = True
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
    reader = _get_ocr_reader()
    if reader is None:
        return []  # OCR no disponible en este despliegue
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

def validate_odf_generico(detections: list) -> dict:
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

    detections = _run_inference(model_key, image, conf)

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

@app.get("/debug-model/{model_key}")
def debug_model(model_key: str):
    """Carga el modelo y devuelve metadata (sin inferencia). Para diagnóstico."""
    import traceback, gc
    if model_key not in MODEL_FILES:
        raise HTTPException(status_code=404, detail=f"Modelo desconocido: {model_key}")
    try:
        session, class_names, input_name, imgsz = _get_model(model_key)
        inp  = session.get_inputs()[0]
        out  = session.get_outputs()[0]
        return {
            "modelo":       model_key,
            "archivo":      MODEL_FILES[model_key],
            "input_name":   input_name,
            "input_shape":  list(inp.shape),
            "output_name":  out.name,
            "output_shape": list(out.shape),
            "class_names":  class_names,
            "imgsz":        imgsz,
            "status":       "ok",
        }
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{tb}")


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


# Endpoint compatible con Telconet (mismo formato URL + respuesta)
@app.post("/predict/fibra/{tn_endpoint}/json")
async def predict_telconet_compat(
    tn_endpoint: str,
    confidence:  float      = 0.25,
    file:        UploadFile = File(...),
):
    import traceback
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

    try:
        r = _run_model(model_key, img, confidence)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] predict {model_key}: {e}\n{tb}", flush=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{tb}")

    return {
        "detections_count": r["count"],
        "detections":       r["detections"],
        "_source":          "render-backup",
        "_model_key":       model_key,
    }


class PredictRequest(BaseModel):
    image_base64: str
    model:        str       = "etiquetas"
    conf:         float     = 0.25
    gis_nombres:  List[str] = []

@app.post("/predict")
def predict(req: PredictRequest):
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
