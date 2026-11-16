"""
Serie temporal (Hito 2 - R3): vida útil restante (RUL) de motores turbofán, NASA C-MAPSS FD001
==============================================================================================
Compara, sobre los 100 motores de prueba:
  0. Alternativa simple: predecir siempre la RUL promedio del entrenamiento.
  1. Modelo clásico: Random Forest sobre resúmenes de la ventana (último valor, media, pendiente).
  2. Deep Learning: LSTM de 2 capas sobre la ventana de 30 ciclos de 14 sensores.

Decisiones estándar en la literatura de C-MAPSS:
  - Se descartan los 7 sensores que no varían en FD001 (una sola condición de operación).
  - La RUL objetivo se recorta en 125 ciclos: al inicio el motor está sano y no hay señal de desgaste.
  - Escalado min-max ajustado SOLO con entrenamiento.
  - Validación separando MOTORES completos (no ciclos) para no filtrar información.

Salidas (carpeta salida/): métricas en JSON, LSTM en ONNX, parámetros de escalado
y las series de prueba que usa el servicio para responder la RUL de cada motor.
"""
import json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DATOS, SAL = "datos", "salida"
os.makedirs(SAL, exist_ok=True)

COLS = ["motor", "ciclo", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSORES = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
VENTANA, TOPE = 30, 125


def leer(nombre):
    return pd.read_csv(f"{DATOS}/{nombre}", sep=r"\s+", header=None, names=COLS)


train, test = leer("train_FD001.txt"), leer("test_FD001.txt")
rul_real = pd.read_csv(f"{DATOS}/RUL_FD001.txt", header=None)[0].values  # RUL verdadera al final de cada motor de prueba

# RUL de entrenamiento: ciclos que faltan hasta la falla, recortada en TOPE
train["rul"] = (train.groupby("motor")["ciclo"].transform("max") - train["ciclo"]).clip(upper=TOPE)

# Escalado min-max con estadísticos de entrenamiento
mn, mx = train[SENSORES].min(), train[SENSORES].max()
esc = lambda df: ((df[SENSORES] - mn) / (mx - mn)).values.astype(np.float32)


def ventanas_entrenamiento(df):
    """Todas las ventanas deslizantes de cada motor, con la RUL del último ciclo como objetivo."""
    X, y = [], []
    for _, g in df.groupby("motor"):
        a, r = esc(g), g["rul"].values
        for fin in range(VENTANA, len(g) + 1):
            X.append(a[fin - VENTANA:fin]); y.append(r[fin - 1])
    return np.stack(X), np.array(y, dtype=np.float32)


def ultima_ventana(g):
    """Última ventana de un motor; si tiene menos de VENTANA ciclos se rellena repitiendo el primero."""
    a = esc(g)
    if len(a) < VENTANA:
        a = np.vstack([np.repeat(a[:1], VENTANA - len(a), 0), a])
    return a[-VENTANA:]


def resumen(X):
    """Features para el modelo clásico: último valor, media y pendiente de cada sensor en la ventana."""
    t = np.arange(VENTANA) - (VENTANA - 1) / 2
    pend = (X * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.hstack([X[:, -1, :], X.mean(1), pend])


def metricas(y, p):
    d = p - y
    score = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))  # puntaje NASA: castiga más predecir tarde
    return {"RMSE": round(float(np.sqrt((d ** 2).mean())), 2), "MAE": round(float(np.abs(d).mean()), 2),
            "score_NASA": round(float(score), 1),
            "sobreestima_mas_de_20": int((d > 20).sum())}  # el error peligroso: creer que le queda más vida


# --- División por motores: 80 entrenamiento / 20 validación
motores = train["motor"].unique(); np.random.shuffle(motores)
val_ids = set(motores[:20])
Xtr, ytr = ventanas_entrenamiento(train[~train["motor"].isin(val_ids)])
Xva, yva = ventanas_entrenamiento(train[train["motor"].isin(val_ids)])
Xte = np.stack([ultima_ventana(g) for _, g in test.groupby("motor")])
yte = np.minimum(rul_real, TOPE).astype(np.float32)  # misma escala recortada que el entrenamiento
print("ventanas", Xtr.shape, Xva.shape, Xte.shape)

res = {}
# 0. Alternativa simple
media = float(ytr.mean())
res["Promedio (simple)"] = metricas(yte, np.full_like(yte, media))

# 1. Random Forest
t0 = time.time()
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, n_jobs=-1, random_state=SEED)
rf.fit(resumen(Xtr), ytr)
p_rf = rf.predict(resumen(Xte))
res["Random Forest (clásico)"] = {**metricas(yte, p_rf), "val_RMSE": metricas(yva, rf.predict(resumen(Xva)))["RMSE"],
                                  "seg_entrenamiento": round(time.time() - t0, 1)}


# 2. LSTM
class LSTM(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.lstm = nn.LSTM(n, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.cab = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.cab(h[:, -1]).squeeze(-1)


t0 = time.time()
m = LSTM(len(SENSORES)); opt = torch.optim.Adam(m.parameters(), 1e-3)
Xt, yt = torch.tensor(Xtr), torch.tensor(ytr / TOPE)  # objetivo normalizado a [0, 1]: la red aprende mucho mejor
Xv = torch.tensor(Xva)
mejor, estado, paciencia, hist = 1e9, None, 0, []
for ep in range(80):
    m.train(); perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 256):
        b = perm[i:i + 256]
        opt.zero_grad(); loss = ((m(Xt[b]) - yt[b]) ** 2).mean(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        rmse_v = float(((m(Xv) * TOPE - torch.tensor(yva)) ** 2).mean().sqrt())
    hist.append(round(rmse_v, 2))
    if rmse_v < mejor - 0.05:
        mejor, estado, paciencia = rmse_v, {k: v.clone() for k, v in m.state_dict().items()}, 0
    else:
        paciencia += 1
        if paciencia >= 12: break  # parada temprana por validación
m.load_state_dict(estado); m.eval()
with torch.no_grad():
    p_lstm = m(torch.tensor(Xte)).numpy() * TOPE  # de vuelta a ciclos
res["LSTM (deep learning)"] = {**metricas(yte, p_lstm), "val_RMSE": round(mejor, 2), "epocas": len(hist),
                               "seg_entrenamiento": round(time.time() - t0, 1)}
print(json.dumps(res, indent=2, ensure_ascii=False))

# --- Exportar LSTM a ONNX y lo que necesita el servicio
class Escalado(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base
    def forward(self, x):
        return self.base(x) * TOPE  # el ONNX entrega la RUL directamente en ciclos


torch.onnx.export(Escalado(m).eval(), torch.tensor(Xte[:1]), f"{SAL}/modelo_rul.onnx", input_names=["ventana"], output_names=["rul"],
                  dynamic_axes={"ventana": {0: "n"}, "rul": {0: "n"}}, opset_version=17, dynamo=False)
series = {int(k): esc(g).round(4).tolist() for k, g in test.groupby("motor")}
json.dump({"sensores": SENSORES, "ventana": VENTANA, "tope": TOPE, "rul_media_train": round(media, 1),
           "min": mn.round(6).tolist(), "max": mx.round(6).tolist()}, open(f"{SAL}/rul_meta.json", "w"), indent=1)
json.dump({"series_escaladas": series, "rul_real": {i + 1: int(r) for i, r in enumerate(rul_real)}},
          open(f"{SAL}/motores_prueba.json", "w"))
# Predicciones por motor (para el tablero) y métricas
pd.DataFrame({"motor": range(1, 101), "rul_real": rul_real, "rul_real_recortada": yte,
              "pred_promedio": round(media, 1), "pred_rf": p_rf.round(1), "pred_lstm": p_lstm.round(1)}
             ).to_csv(f"{SAL}/predicciones_prueba.csv", index=False)
json.dump({"metricas_prueba": res, "historial_val_RMSE_lstm": hist}, open(f"{SAL}/metricas_rul.json", "w"),
          indent=2, ensure_ascii=False)

# --- Random Forest a ONNX (para que el servicio no necesite scikit-learn) y combinación conservadora
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
onx = convert_sklearn(rf, initial_types=[("x", FloatTensorType([None, 3 * len(SENSORES)]))], target_opset=17)
open(f"{SAL}/modelo_rul_rf.onnx", "wb").write(onx.SerializeToString())
# Regla operativa: usar la MENOR de las dos predicciones (sobreestimar la vida es el error caro)
p_min = np.minimum(p_rf, p_lstm)
res["Mínimo RF/LSTM (operativo)"] = metricas(yte, p_min)
json.dump({"metricas_prueba": res, "historial_val_RMSE_lstm": hist}, open(f"{SAL}/metricas_rul.json", "w"),
          indent=2, ensure_ascii=False)
print("operativo", res["Mínimo RF/LSTM (operativo)"])
