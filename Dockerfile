# Servicio de inspección (Render, plan gratis). Imagen liviana: sin torch, solo onnxruntime.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Render entrega el puerto en la variable PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
