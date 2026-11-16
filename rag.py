"""
RAG sobre el manual de mantenimiento - Hito 2, R5 (Transformer)
================================================================
Búsqueda semántica con embeddings de all-MiniLM-L6-v2 (Transformer de 6 capas, 22 M de parámetros,
basado en atención), ejecutado en ONNX. El índice se arma una vez (indexar.py, al construir el
contenedor): el manual se corta en fragmentos con su página, cada fragmento se convierte en un vector
de 384 dimensiones y se guarda. En cada consulta se codifica la pregunta y se devuelven los fragmentos
más parecidos (similitud coseno) con su fuente y página, para que el LLM responda citándolos.
"""
import json
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

DIR_MODELO = os.getenv("DIR_EMBEDDINGS", "embeddings")
SIM_MINIMA = float(os.getenv("SIM_MINIMA", "0.25"))  # bajo esto el fragmento no se considera relevante


class Embedder:
    """Codificador de oraciones: tokeniza, pasa por el Transformer y promedia los tokens (mean pooling)."""

    def __init__(self, carpeta=DIR_MODELO):
        self.tok = Tokenizer.from_file(f"{carpeta}/tokenizer.json")
        self.tok.enable_truncation(max_length=256)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self.ses = ort.InferenceSession(f"{carpeta}/model.onnx", providers=["CPUExecutionProvider"])
        self.entradas = {i.name for i in self.ses.get_inputs()}

    def __call__(self, textos, lote=32):
        out = []
        for i in range(0, len(textos), lote):
            enc = self.tok.encode_batch(textos[i:i + lote])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.entradas:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.ses.run(None, feed)[0]                       # (lote, tokens, 384)
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)    # promedio sobre tokens reales
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))  # normalizado: producto = coseno
        return np.vstack(out).astype(np.float32)


class Indice:
    def __init__(self, carpeta="indice"):
        self.ok = os.path.exists(f"{carpeta}/vectores.npy")
        if not self.ok:
            return
        self.vec = np.load(f"{carpeta}/vectores.npy")
        self.frag = json.load(open(f"{carpeta}/fragmentos.json", encoding="utf-8"))
        self.emb = Embedder()

    def buscar(self, pregunta: str, k: int = 3) -> list[dict]:
        q = self.emb([pregunta])[0]
        sims = self.vec @ q
        res = []
        for i in np.argsort(-sims)[:k]:
            if sims[i] < SIM_MINIMA:
                break
            f = self.frag[i]
            res.append({"fuente": f["fuente"], "pagina": f["pagina"], "similitud": round(float(sims[i]), 3),
                        "texto": f["texto"]})
        return res
