from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
class RateResponse(BaseModel):
    rate: float
# ---------------------------------------------------------------------------
# Configuration: adjust filenames if different
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.getenv('MODEL_FILE', 'my_model_improved.h5')
SCALER_FILE = os.getenv('SCALER_FILE', 'target_scaler_improved.pkl')
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)
SCALER_PATH = os.path.join(BASE_DIR, SCALER_FILE)
# Default sequence length used during model training; override via env var if needed
DEFAULT_SEQ_LEN = 15
SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH', DEFAULT_SEQ_LEN))

# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Land Price Prediction Service",
    description="Predicts land price per m² from a feature vector.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # or ["*"] for development
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    # The raw feature vector used for prediction
    features: List[float]

class PredictionResponse(BaseModel):
    predicted_price: float

# ---------------------------------------------------------------------------
# Load model and scaler at startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global model, target_scaler, input_dim, SEQUENCE_LENGTH
    # Load the Keras model for inference (no compile)
    try:
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{MODEL_PATH}': {e}")
    # Load the target scaler (must be a joblib pickle)
    try:
        target_scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler '{SCALER_PATH}': {e}")
    # Infer the expected input feature dimension from the model
    shape = model.input_shape  # e.g. (None, sequence_length, feature_dim)
    if len(shape) != 3:
        raise RuntimeError(f"Unexpected model input shape: {shape}")
    # Adjust sequence length if it differs
    if shape[1] != SEQUENCE_LENGTH:
        SEQUENCE_LENGTH = shape[1]
    input_dim = shape[2]

# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    # Validate input length
    if len(req.features) != input_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 'features' list of length {input_dim}, got {len(req.features)}"
        )
    try:
        # Build feature array [1, input_dim]
        x = np.array(req.features, dtype=float).reshape(1, input_dim)
        # Repeat along sequence axis -> [1, SEQUENCE_LENGTH, input_dim]
        x_seq = np.tile(x, (SEQUENCE_LENGTH, 1)).reshape(1, SEQUENCE_LENGTH, input_dim)
        # Get model output (scaled)
        y_scaled = model.predict(x_seq)
        # Inverse-transform to original scale
        y_pred = target_scaler.inverse_transform(y_scaled).flatten()[0]
        return PredictionResponse(predicted_price=float(y_pred))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ---------------------------------------------------------------------------
@app.get("/rate", response_model=RateResponse)
def get_usd_to_tnd_rate():
    # external API (no key needed)
    url = "https://api.exchangerate.host/latest?base=USD&symbols=TND"
    resp = requests.get(url, timeout=5)
    if resp.status_code != 200:
        raise HTTPException(502, "Failed to fetch exchange rate")
    data = resp.json()
    rate = data.get("rates", {}).get("TND")
    if rate is None:
        raise HTTPException(502, "Malformed rate response")
    return RateResponse(rate=rate)

# ---------------------------------------------------------------------------
# Health-check endpoint
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "running"}

# ---------------------------------------------------------------------------
# Uvicorn entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # If this file is named 'class.py', use "class:app"; else adjust to match filename
    uvicorn.run("class:app", host="0.0.0.0", port=4000, reload=True)
