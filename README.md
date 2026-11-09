# Inspección de álabes AeBAD (Hito 1)

API en FastAPI que clasifica fotos de álabes de motor con EfficientNet-B0 (fine-tuning sobre AeBAD, exportado a ONNX)
y sugiere una decisión: liberar, reinspeccionar o retirar. La llama un flujo de n8n.

- `GET /health`: estado del servicio y clases del modelo
- `POST /predict`: imagen en multipart, campo `file`
- `GET /docs`: documentación interactiva

Despliegue: Render (Docker, plan gratis). `convertir_onnx.py` muestra cómo se exportó el modelo desde PyTorch/timm.
