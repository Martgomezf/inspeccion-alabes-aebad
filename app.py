"""
Servicio de inspección de álabes (Hito 1 - R2 y R4)
====================================================
API en FastAPI que se despliega en Render (plan gratis) y que n8n llama por HTTP.

Qué hace (lógica propia, no solo copiar datos):
  1. Control de calidad de la foto (brillo y nitidez) antes de confiar en el modelo.
  2. Clasificación con EfficientNet-B0 preentrenado en ImageNet y ajustado (fine-tuning) con AeBAD.
  3. Regla de decisión asimétrica: dejar pasar un defecto ("escape") cuesta mucho más
     que una falsa alarma, por eso el umbral para pedir reinspección es bajo.
  4. Indicadores para el LLM y el registro: prob. de defecto, confianza, costo esperado.

Endpoints:
  GET  /health   -> estado del servicio y del modelo
  POST /predict  -> recibe una imagen (multipart, campo "file") y devuelve JSON
"""

import io
import json
import os
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Configuración (se puede cambiar con variables de entorno en Render)
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_aebad.onnx")  # EfficientNet-B0 exportado a ONNX
META_PATH = os.getenv("META_PATH", "modelo_meta.json")     # clases, tamaño y normalización
GOOD_LABEL = os.getenv("GOOD_LABEL", "")                   # clase "sin defecto" (vacío = detectar sola)
NOMBRES_SANOS = ["good", "normal", "ok", "sin_defecto", "sin defecto", "bueno", "sano", "no_defect"]

# Umbrales sobre la probabilidad de defecto (1 - P(good)).
# Bajos a propósito: preferimos reinspeccionar de más antes que liberar un álabe dañado.
UMBRAL_REINSPECCION = float(os.getenv("UMBRAL_REINSPECCION", "0.25"))
UMBRAL_RETIRO = float(os.getenv("UMBRAL_RETIRO", "0.60"))

# Costos supuestos (USD) para el indicador de negocio (R7). Ajustar con supuestos del informe.
COSTO_ESCAPE = float(os.getenv("COSTO_ESCAPE", "50000"))        # defecto liberado que falla en servicio
COSTO_REINSPECCION = float(os.getenv("COSTO_REINSPECCION", "120"))  # hora de inspector + boroscopio
COSTO_RETIRO_INNECESARIO = float(os.getenv("COSTO_RETIRO_INNECESARIO", "3000"))  # álabe bueno descartado

# Control de calidad de imagen
BRILLO_MIN, BRILLO_MAX = 25, 235   # media de gris (0-255)
NITIDEZ_MIN = 15.0                 # varianza del laplaciano; bajo = foto movida/desenfocada

# ---------------------------------------------------------------------------
# Carga del modelo (ONNX: no necesita torch, cabe en el plan gratis de Render)
# ---------------------------------------------------------------------------
with open(META_PATH, encoding="utf-8") as f:
    META = json.load(f)
ETIQUETAS = META["clases"]
IMG = int(META["img_size"])
MEDIA = np.array(META["media"], dtype=np.float32).reshape(3, 1, 1)
STD = np.array(META["std"], dtype=np.float32).reshape(3, 1, 1)

try:
    SESION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    ENTRADA = SESION.get_inputs()[0].name
    ESTADO_MODELO = "ok"
except Exception as e:  # el servicio igual levanta y /health muestra el problema
    SESION, ESTADO_MODELO = None, f"error al cargar: {e}"

# Clase "sana": la indicada en GOOD_LABEL o la primera que calce con un nombre típico
if not GOOD_LABEL:
    GOOD_LABEL = next((c for c in ETIQUETAS if c.lower().strip() in NOMBRES_SANOS), ETIQUETAS[-1])


def preprocesar(img: Image.Image) -> np.ndarray:
    """Igual que en el entrenamiento: redimensionar a IMG x IMG y normalizar con media/std de ImageNet."""
    x = np.asarray(img.convert("RGB").resize((IMG, IMG), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x.transpose(2, 0, 1) - MEDIA) / STD          # HWC -> CHW
    return x[None].astype(np.float32)                  # agrega dimensión de lote


# ---------------------------------------------------------------------------
# Lógica propia
# ---------------------------------------------------------------------------
def calidad_imagen(img: Image.Image) -> dict:
    """Brillo medio y nitidez (varianza del laplaciano) calculados con numpy."""
    gris = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
    brillo = float(gris.mean())
    # Laplaciano discreto 4-vecinos sin depender de OpenCV
    lap = (-4 * gris[1:-1, 1:-1] + gris[:-2, 1:-1] + gris[2:, 1:-1]
           + gris[1:-1, :-2] + gris[1:-1, 2:])
    nitidez = float(lap.var())
    problemas = []
    if brillo < BRILLO_MIN:
        problemas.append("imagen muy oscura")
    if brillo > BRILLO_MAX:
        problemas.append("imagen sobreexpuesta")
    if nitidez < NITIDEZ_MIN:
        problemas.append("imagen desenfocada o movida")
    return {"brillo": round(brillo, 1), "nitidez": round(nitidez, 1),
            "apta": not problemas, "problemas": problemas}


def predecir(img: Image.Image) -> dict:
    """Probabilidades por clase (softmax sobre los logits del modelo)."""
    logits = SESION.run(None, {ENTRADA: preprocesar(img)})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return {etq: float(p) for etq, p in zip(ETIQUETAS, probs)}


def decidir(probs: dict, calidad: dict) -> dict:
    """Regla de decisión asimétrica + costo esperado de cada alternativa."""
    p_defecto = 1.0 - probs.get(GOOD_LABEL, 0.0)
    clase_top = max(probs, key=probs.get)
    confianza = probs[clase_top]

    if not calidad["apta"]:
        decision = "reinspeccionar"
        motivo = "foto no apta: " + ", ".join(calidad["problemas"])
    elif p_defecto >= UMBRAL_RETIRO:
        decision, motivo = "retirar", "probabilidad de defecto alta"
    elif p_defecto >= UMBRAL_REINSPECCION:
        decision, motivo = "reinspeccionar", "zona de incertidumbre"
    else:
        decision, motivo = "liberar", "probabilidad de defecto baja"

    # Costo esperado (USD) de cada acción dado p_defecto -> muestra por qué el umbral es bajo
    costo_esperado = {
        "liberar": round(p_defecto * COSTO_ESCAPE, 0),
        "reinspeccionar": round(COSTO_REINSPECCION, 0),
        "retirar": round((1 - p_defecto) * COSTO_RETIRO_INNECESARIO, 0),
    }
    requiere_humano = decision != "liberar" or confianza < 0.70

    # Tipo de defecto más probable (si el modelo es multiclase)
    defectos = {k: v for k, v in probs.items() if k != GOOD_LABEL}
    tipo = max(defectos, key=defectos.get) if defectos else None

    return {
        "decision_sugerida": decision,
        "motivo_regla": motivo,
        "p_defecto": round(p_defecto, 4),
        "clase_mas_probable": clase_top,
        "tipo_defecto_probable": tipo,
        "confianza": round(confianza, 4),
        "requiere_revision_humana": requiere_humano,
        "costo_esperado_usd": costo_esperado,
    }

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="Inspección de álabes AeBAD")


@app.get("/health")
def health():
    return {"servicio": "ok", "modelo": ESTADO_MODELO, "clases": ETIQUETAS,
            "clase_sana": GOOD_LABEL, "img_size": IMG,
            "umbrales": {"reinspeccion": UMBRAL_REINSPECCION, "retiro": UMBRAL_RETIRO}}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if SESION is None:
        raise HTTPException(503, f"Modelo no disponible ({ESTADO_MODELO})")
    t0 = time.time()
    contenido = await file.read()
    try:
        img = Image.open(io.BytesIO(contenido))
        img.load()
    except (UnidentifiedImageError, OSError):
        # n8n recibe un 422 y el flujo de errores avisa por Telegram
        raise HTTPException(422, f"El archivo '{file.filename}' no es una imagen válida")

    calidad = calidad_imagen(img)
    probs = predecir(img)
    resultado = decidir(probs, calidad)

    return {
        "archivo": file.filename,
        "modelo": "EfficientNet-B0 (fine-tuning AeBAD, ONNX)",
        "probabilidades": {k: round(v, 4) for k, v in probs.items()},
        "calidad_imagen": calidad,
        **resultado,
        "tiempo_ms": round((time.time() - t0) * 1000),
    }
