import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI(title="SkinD | Skin Disease Predictor")

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
# Using the .keras format which is more modern
MODEL_PATH = "skin_model.keras"
if not os.path.exists(MODEL_PATH):
    # Fallback to .h5 if .keras doesn't exist
    MODEL_PATH = "best_skin_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class names exactly as in the notebook
class_names = [
    'ch_Chickenpox_Varicela',
    'hz_Herpes',
    'lp_Lupus',
    'me_Melanoma',
    'mk_Monkeypox',
    'ms_Measles_Sarampion',
    'sc_Scabies_sarna'
]

# Human-readable labels
human_labels = {
    'ch_Chickenpox_Varicela': 'Chickenpox (Varicella)',
    'hz_Herpes': 'Herpes',
    'lp_Lupus': 'Lupus',
    'me_Melanoma': 'Melanoma (Skin Cancer)',
    'mk_Monkeypox': 'Monkeypox',
    'ms_Measles_Sarampion': 'Measles',
    'sc_Scabies_sarna': 'Scabies'
}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(content={"error": "Model not loaded"}, status_code=500)
    
    try:
        content = await file.read()
        processed_image = preprocess_image(content)
        
        predictions = model.predict(processed_image)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        
        results = []
        for idx in top_indices:
            name = class_names[idx]
            results.append({
                "class": human_labels.get(name, name),
                "confidence": float(predictions[0][idx]) * 100
            })
            
        return {
            "prediction": results[0]["class"],
            "confidence": results[0]["confidence"],
            "all_predictions": results
        }
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Serve static files (HTML, CSS, JS)
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Using 127.0.0.1 specifically for Windows local access
    print("\n" + "="*50)
    print("SERVERS STARTING... PLEASE WAIT 10 SECONDS")
    print("URL: http://127.0.0.1:8080")
    print("="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8080)
