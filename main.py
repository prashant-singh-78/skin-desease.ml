import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI(title="Skin Disease Predictor")

# ✅ CORS (frontend connect karega)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Model Load
MODEL_PATH = "skin_model.keras"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "best_skin_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded")
except Exception as e:
    print("❌ Model Error:", e)
    model = None

# ✅ Classes
class_names = [
    'ch_Chickenpox_Varicela',
    'hz_Herpes',
    'lp_Lupus',
    'me_Melanoma',
    'mk_Monkeypox',
    'ms_Measles_Sarampion',
    'sc_Scabies_sarna'
]

human_labels = {
    'ch_Chickenpox_Varicela': 'Chickenpox',
    'hz_Herpes': 'Herpes',
    'lp_Lupus': 'Lupus',
    'me_Melanoma': 'Melanoma',
    'mk_Monkeypox': 'Monkeypox',
    'ms_Measles_Sarampion': 'Measles',
    'sc_Scabies_sarna': 'Scabies'
}

# ✅ Image preprocess
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(content={"error": "Model not loaded"}, status_code=500)

    try:
        content = await file.read()
        img = preprocess_image(content)
        preds = model.predict(img)

        idx = np.argmax(preds[0])
        return {
            "prediction": human_labels[class_names[idx]],
            "confidence": float(preds[0][idx]) * 100
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ✅ Static frontend serve karega
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# ❌ Flask wala part hata diya (important)
