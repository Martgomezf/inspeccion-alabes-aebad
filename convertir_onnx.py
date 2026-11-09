# Convierte el paquete de Colab (timm EfficientNet-B0) a ONNX y guarda los metadatos aparte
import json, sys, torch, timm
src, dst = sys.argv[1], sys.argv[2]
p = torch.load(src, map_location="cpu", weights_only=False)
m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(p["clases"]))
m.load_state_dict(p["state_dict"]); m.eval()
x = torch.randn(1, 3, p["img_size"], p["img_size"])
torch.onnx.export(m, x, f"{dst}/modelo_aebad.onnx", input_names=["imagen"], output_names=["logits"],
                  dynamic_axes={"imagen": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
json.dump({k: p[k] for k in ["arquitectura", "clases", "img_size", "media", "std"]},
          open(f"{dst}/modelo_meta.json", "w"), indent=2)
# Verificación: misma salida en torch y en onnxruntime
import onnxruntime as ort, numpy as np
s = ort.InferenceSession(f"{dst}/modelo_aebad.onnx")
d = np.abs(s.run(None, {"imagen": x.numpy()})[0] - m(x).detach().numpy()).max()
print("diferencia máx torch vs onnx:", d)
