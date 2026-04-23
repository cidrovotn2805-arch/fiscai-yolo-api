---
title: FiscAI YOLO API
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# FiscAI YOLO API — Detección visual para fiscalización de mangas

API FastAPI con dos modelos YOLO para validación de fotos de mangas de empalme.

## Modelos

| Modelo | Clases | Foto |
|--------|--------|------|
| `etiquetas` | ETIQUETA 1, ETIQUETA 2, MANGA | Foto 2 — Etiquetas en FO al ingreso |
| `manga`     | Manga, Seguros 1, Seguros 2, Tapones | Foto 1 — Manga correctamente sellada |

## Endpoints

### `GET /health`
```json
{"status": "ok", "models": {"etiquetas": [...], "manga": [...]}}
```

### `POST /predict`
```json
{
  "image_base64": "<base64 sin prefijo data:...>",
  "model": "manga",
  "conf": 0.25
}
```

**Respuesta modelo `manga`:**
```json
{
  "model": "manga",
  "detections": [...],
  "classes_found": ["Manga", "Seguros 1", "Tapones"],
  "validation": {
    "manga_presente": true,
    "seguros_presentes": true,
    "seguro1": true,
    "seguro2": false,
    "tapones_presentes": true,
    "aprobado": true
  }
}
```

**Respuesta modelo `etiquetas`:**
```json
{
  "model": "etiquetas",
  "validation": {
    "etiqueta1_presente": true,
    "etiqueta2_presente": true,
    "aprobado": true
  }
}
```
