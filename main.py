# src/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- CORS MIDDLEWARE SETUP ---
# This is the crucial part that allows your React app to talk to this server.
origins = [
    "http://localhost:5173",  # The address of your React frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# -----------------------------

# --- MODEL LOADING ---
try:
    MODEL = tf.keras.models.load_model("models/1.keras")
    CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    MODEL = None

@app.get("/ping")
async def ping():
    return "Hello, I am alive..."

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs.")
    
    try:
        # 1. Read the image from the uploaded file
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # 2. Convert the image to a NumPy array.
        #    DO NOT resize or normalize here.
        image_array = np.array(image)

        # 3. Add a batch dimension
        img_batch = np.expand_dims(image_array, 0)
        
        # 4. Make prediction. The model will handle the rest internally.
        predictions = MODEL.predict(img_batch)
        
        # 5. Process the result
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            'class': predicted_class,
            'confidence': confidence
        }

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)