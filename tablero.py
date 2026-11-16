"""
Tablero (Hito 2 - R9): página web pública servida por el mismo servicio en /tablero.
Lee en vivo la hoja "registro" de Google Sheets (compartida como "cualquiera con el enlace puede ver")
y la combina con la RUL de la flota y las métricas de los modelos. /tablero/datos entrega el JSON.
"""
import csv
import io
import json
import os
import time
import urllib.request

import rul as modulo_rul

SHEET_ID = os.getenv("SHEET_ID", "1A94avOmZ1PeLLiPf0DJQonD9Gf5JFOQIEw5YtJuoE7k")
HOJA = os.getenv("SHEET_HOJA", "registro")
_cache = {"t": 0, "flota": None}

METRICAS = {
    "imagenes": [  # prueba, 324 imágenes AeBAD-S (entrega Hito 1)
        {"modelo": "Siempre good", "f1_macro": 12.8, "escapes": 172},
        {"modelo": "CLIP sin entrenar", "f1_macro": 31.6, "escapes": None},
        {"modelo": "ResNet-50", "f1_macro": 88.7, "escapes": 15},
        {"modelo": "EfficientNet-B0", "f1_macro": 84.5, "escapes": 11}],
    "rul": json.load(open("metricas_rul.json", encoding="utf-8"))["metricas_prueba"] if os.path.exists("metricas_rul.json") else {},
}


def leer_registro():
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={HOJA}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            texto = r.read().decode("utf-8")
        if texto.lstrip().startswith("<"):
            return [], "La hoja no es pública: compártela como 'cualquiera con el enlace puede ver'"
        return list(csv.DictReader(io.StringIO(texto))), None
    except Exception as e:
        return [], f"No se pudo leer la hoja: {e}"


def flota():
    """RUL operativa de los 100 motores (se recalcula cada 10 minutos)."""
    if _cache["flota"] is None or time.time() - _cache["t"] > 600:
        f = []
        for m in range(1, 101):
            r = modulo_rul.predecir_rul(m)
            f.append({"motor": m, "rul": r["rul_operativa"], "estado": r["estado_motor"]})
        _cache.update(t=time.time(), flota=f)
    return _cache["flota"]


def datos():
    filas, error = leer_registro()
    return {"registro": filas, "error_registro": error, "flota": flota(), "metricas": METRICAS,
            "generado": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}


HTML = open("tablero.html", encoding="utf-8").read() if os.path.exists("tablero.html") else "<p>sin tablero</p>"
