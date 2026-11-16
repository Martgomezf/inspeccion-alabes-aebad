"""
Construye el índice del RAG (se ejecuta al construir el contenedor en Render).
1. Descarga all-MiniLM-L6-v2 en ONNX desde Hugging Face (si no está).
2. Lee los PDF de docs/, corta cada página en fragmentos de ~120 palabras con 30 de traslape.
3. Calcula los embeddings y guarda indice/vectores.npy + indice/fragmentos.json.
"""
import glob
import json
import os
import re

import numpy as np
from pypdf import PdfReader

MODELO = "sentence-transformers/all-MiniLM-L6-v2"
os.makedirs("embeddings", exist_ok=True)
if not os.path.exists("embeddings/model.onnx"):
    from huggingface_hub import hf_hub_download
    import shutil
    shutil.copy(hf_hub_download(MODELO, "onnx/model.onnx"), "embeddings/model.onnx")
    shutil.copy(hf_hub_download(MODELO, "tokenizer.json"), "embeddings/tokenizer.json")

from rag import Embedder  # noqa: E402  (necesita el modelo ya descargado)

PALABRAS, TRASLAPE = 120, 30
frag = []
for ruta in sorted(glob.glob("docs/*.pdf")):
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    for n, pag in enumerate(PdfReader(ruta).pages, start=1):
        texto = re.sub(r"\s+", " ", pag.extract_text() or "").strip()
        pal = texto.split()
        if len(pal) < 25:      # páginas casi vacías (índices, en blanco) no aportan
            continue
        for i in range(0, len(pal), PALABRAS - TRASLAPE):
            trozo = " ".join(pal[i:i + PALABRAS])
            if len(trozo.split()) >= 25:
                frag.append({"fuente": nombre, "pagina": n, "texto": trozo})

# Documentos ya extraídos a texto por página (JSON): {"fuente": ..., "paginas": [{"pagina": "10-52", "texto": ...}]}
# Se usa la numeración impresa del manual para que la cita se pueda verificar en el libro.
for ruta in sorted(glob.glob("docs/*.json") + glob.glob("*_manual.json")):
    doc = json.load(open(ruta, encoding="utf-8"))
    for p in doc["paginas"]:
        pal = p["texto"].split()
        for i in range(0, len(pal), PALABRAS - TRASLAPE):
            trozo = " ".join(pal[i:i + PALABRAS])
            if len(trozo.split()) >= 25:
                frag.append({"fuente": doc["fuente"], "pagina": p["pagina"], "texto": trozo})

print(f"{len(frag)} fragmentos de {len(glob.glob('docs/*.pdf'))} documentos")
vec = Embedder()([f["texto"] for f in frag])
os.makedirs("indice", exist_ok=True)
np.save("indice/vectores.npy", vec)
json.dump(frag, open("indice/fragmentos.json", "w", encoding="utf-8"), ensure_ascii=False)
print("índice listo:", vec.shape)
