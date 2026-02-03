from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

# =====================================
# LOAD MODEL
# =====================================
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

CLASS_NAMES = model.names  # {0: 'paregi', 1: 'pelise'}
IMG_SIZE = 640
CONF_THRESH = 0.15

print("✅ YOLOv8 model loaded")

# =====================================
# FASTAPI APP
# =====================================
app = FastAPI(title="Paregi / Pelise YOLOv8 API")

# =====================================
# PREDICTION ENDPOINT
# =====================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        verbose=False
    )

    r = results[0]

    if len(r.boxes) == 0:
        return {
            "prediction": "no_object_detected",
            "confidence": 0.0,
            "probabilities": {
                CLASS_NAMES[0]: 0.0,
                CLASS_NAMES[1]: 0.0
            }
        }

    idx = r.boxes.conf.argmax()
    cls_id = int(r.boxes.cls[idx])
    conf = float(r.boxes.conf[idx])

    probs = {
        CLASS_NAMES[0]: 0.0,
        CLASS_NAMES[1]: 0.0
    }
    probs[CLASS_NAMES[cls_id]] = round(conf, 4)

    return {
        "prediction": CLASS_NAMES[cls_id],
        "confidence": round(conf, 4),
        "probabilities": probs
    }

# =====================================
# HEALTH CHECK (RENDER NEEDS THIS)
# =====================================
@app.get("/")
def health():
    return {"status": "ok"}
