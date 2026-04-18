import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

app = FastAPI(title="Skin Disease Predictor")

# ✅ CORS
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
    print(f"✅ Model Loaded from {MODEL_PATH}")
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

# ✅ Root route (important for Render)
@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}

# ✅ Image preprocess
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ Prediction API
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

# ✅ Optional: static folder (safe way)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Local run only
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
