# Servicio de inspección (Render, plan gratis). Sin torch: los tres modelos corren en onnxruntime.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Construye el índice del RAG: descarga el Transformer de embeddings y codifica el manual de docs/
RUN python indexar.py
# Render entrega el puerto en la variable PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
