"""
Vida útil restante (RUL) del motor - Hito 2, R3
================================================
Responde la RUL de un motor del conjunto de prueba NASA C-MAPSS FD001 con dos modelos
entrenados en serie_temporal.py (ambos exportados a ONNX):
  - LSTM de 2 capas sobre la ventana de 30 ciclos (deep learning)
  - Random Forest sobre resúmenes de la ventana (clásico)
Regla operativa propia: se usa la MENOR de las dos predicciones, porque sobreestimar la vida
que le queda al motor es el error caro (en prueba: RMSE 11,7 y 4 sobreestimaciones > 20 ciclos,
mejor que cada modelo por separado).
"""
import json

import numpy as np
import onnxruntime as ort

META = json.load(open("rul_meta.json"))
MOTORES = json.load(open("motores_prueba.json"))
V = META["ventana"]
SENSORES = META["sensores"]
LSTM = ort.InferenceSession("modelo_rul.onnx", providers=["CPUExecutionProvider"])
RF = ort.InferenceSession("modelo_rul_rf.onnx", providers=["CPUExecutionProvider"])

# Umbrales de estado en ciclos de vuelo restantes (supuestos operativos, ajustables)
CRITICO, ATENCION = 30, 60


def _ventana(a):
    """Últimos V ciclos; si el motor tiene menos, se repite el primero (igual que al entrenar)."""
    a = np.asarray(a, dtype=np.float32)
    if len(a) < V:
        a = np.vstack([np.repeat(a[:1], V - len(a), 0), a])
    return a[-V:]


def _resumen(x):
    """Features del Random Forest: último valor, media y pendiente de cada sensor."""
    t = np.arange(V) - (V - 1) / 2
    pend = (x * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.hstack([x[:, -1, :], x.mean(1), pend]).astype(np.float32)


def predecir_rul(motor: int, hasta_ciclo: int | None = None) -> dict:
    """RUL del motor usando su historia hasta `hasta_ciclo` (por defecto, toda la disponible)."""
    serie = MOTORES["series_escaladas"].get(str(motor))
    if serie is None:
        raise KeyError(f"Motor {motor} no existe (válidos: 1 a {len(MOTORES['series_escaladas'])})")
    total = len(serie)
    n = total if hasta_ciclo is None else max(1, min(hasta_ciclo, total))
    x = _ventana(serie[:n])[None]
    p_lstm = float(LSTM.run(None, {"ventana": x})[0].ravel()[0])
    p_rf = float(RF.run(None, {"x": _resumen(x)})[0].ravel()[0])
    rul = max(0.0, min(p_lstm, p_rf))  # regla conservadora

    # Tendencia: RUL operativa estimada en los últimos 5 puntos de la historia (cada 10 ciclos)
    tendencia = []
    for c in range(max(V, n - 40), n + 1, 10):
        xc = _ventana(serie[:c])[None]
        tendencia.append({"ciclo": c, "rul": round(max(0.0, min(float(LSTM.run(None, {"ventana": xc})[0].ravel()[0]),
                                                               float(RF.run(None, {"x": _resumen(xc)})[0].ravel()[0]))), 1)})

    estado = "critico" if rul <= CRITICO else "atencion" if rul <= ATENCION else "normal"
    real = MOTORES["rul_real"][str(motor)] + (total - n)  # solo para validar en la demo, el modelo no lo usa
    return {
        "motor": motor, "ciclos_observados": n,
        "rul_lstm": round(p_lstm, 1), "rul_random_forest": round(p_rf, 1),
        "rul_operativa": round(rul, 1), "estado_motor": estado,
        "umbrales_ciclos": {"critico": CRITICO, "atencion": ATENCION},
        "tendencia_rul": tendencia,
        "rul_real_referencia": real,
        "modelos": "LSTM 2x64 y Random Forest 100 árboles, NASA C-MAPSS FD001; operativa = mínimo de ambos",
    }
