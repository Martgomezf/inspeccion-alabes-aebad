"""
Comparación del modelo con alternativas simples (para Colab) — Hito 1, criterio R4 (30 %)
==========================================================================================
Compara sobre el MISMO conjunto de prueba:
  0. Línea base trivial: predecir siempre la clase mayoritaria.
  1. Alternativa simple: regresión logística sobre histogramas de color + nitidez
     (sin deep learning).
  2. EfficientNet-B0 preentrenado y ajustado (el modelo del sistema).
  (Opcional) ResNet-50 si tienen sus pesos.

Métricas: exactitud, recall de defectos (lo que importa: defectos que NO se escapan),
precisión, F1 y costo total con los mismos supuestos de costo del servicio.

Estructura esperada (torchvision ImageFolder):
  datos/train/<clase>/*.jpg
  datos/test/<clase>/*.jpg
Uso en Colab:
  !python comparacion_modelos.py --datos /content/datos --pesos model.pth
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, models, transforms

COSTO_ESCAPE, COSTO_FALSA_ALARMA = 50000, 120   # mismos supuestos que app.py


def features_simples(ruta):
    """Histograma RGB (16 bins por canal) + brillo + nitidez: 50 variables."""
    img = Image.open(ruta).convert("RGB").resize((224, 224))
    a = np.asarray(img, dtype=np.float32)
    hist = [np.histogram(a[..., c], bins=16, range=(0, 255), density=True)[0] for c in range(3)]
    g = a.mean(axis=2)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return np.concatenate(hist + [[g.mean() / 255, np.log1p(lap.var())]])


def metricas(nombre, y, yhat, idx_good):
    """Métricas con 'defecto' como clase positiva (todo lo que no es good)."""
    yb, yhb = (y != idx_good).astype(int), (yhat != idx_good).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yb, yhb, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yb, yhb, labels=[0, 1]).ravel()
    return {"modelo": nombre, "exactitud": round(accuracy_score(y, yhat), 3),
            "recall_defecto": round(r, 3), "precision_defecto": round(p, 3), "f1": round(f, 3),
            "defectos_escapados": int(fn), "falsas_alarmas": int(fp),
            "costo_total_usd": int(fn * COSTO_ESCAPE + fp * COSTO_FALSA_ALARMA)}


def predecir_cnn(modelo, ds):
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    modelo.eval()
    preds = []
    with torch.no_grad():
        for ruta, _ in ds.samples:
            x = tf(Image.open(ruta).convert("RGB")).unsqueeze(0)
            preds.append(int(modelo(x).argmax(1)))
    return np.array(preds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--pesos", default="model.pth")
    ap.add_argument("--pesos_resnet", default=None)
    ap.add_argument("--good", default="good")
    args = ap.parse_args()

    train = datasets.ImageFolder(f"{args.datos}/train")
    test = datasets.ImageFolder(f"{args.datos}/test")
    idx_good = test.class_to_idx[args.good]
    y_tr = np.array(train.targets)
    y_te = np.array(test.targets)
    resultados = []

    # 0. Clase mayoritaria
    mayoritaria = np.bincount(y_tr).argmax()
    resultados.append(metricas("Clase mayoritaria", y_te, np.full_like(y_te, mayoritaria), idx_good))

    # 1. Regresión logística sobre features simples
    X_tr = np.stack([features_simples(r) for r, _ in train.samples])
    X_te = np.stack([features_simples(r) for r, _ in test.samples])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(X_tr, y_tr)
    resultados.append(metricas("Reg. logística (histogramas)", y_te, lr.predict(X_te), idx_good))

    # 2. EfficientNet-B0
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(test.classes))
    m.load_state_dict(torch.load(args.pesos, map_location="cpu"))
    resultados.append(metricas("EfficientNet-B0 (fine-tuning)", y_te, predecir_cnn(m, test), idx_good))

    # Opcional: ResNet-50
    if args.pesos_resnet:
        r = models.resnet50(weights=None)
        r.fc = nn.Linear(r.fc.in_features, len(test.classes))
        r.load_state_dict(torch.load(args.pesos_resnet, map_location="cpu"))
        resultados.append(metricas("ResNet-50 (fine-tuning)", y_te, predecir_cnn(r, test), idx_good))

    try:
        import pandas as pd
        print(pd.DataFrame(resultados).to_string(index=False))
    except ImportError:
        print(json.dumps(resultados, indent=2, ensure_ascii=False))
    json.dump(resultados, open("comparacion.json", "w"), indent=2, ensure_ascii=False)
    # Guardar el orden de clases para el Space
    json.dump(test.classes, open("labels.json", "w"))
    print("\nGuardado comparacion.json y labels.json")
