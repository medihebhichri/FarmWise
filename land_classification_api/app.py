#!/usr/bin/env python3
# main.py

import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Crop Recommendation Service",
    description="Given soil features, returns a recommended crop.",
    version="1.0.0"
)
# ─── C O R S ────────────────────────────────────────────────────────────
# allow your Angular dev server to talk to this FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],     # GET, POST, OPTIONS, etc.
    allow_headers=["*"],     # Authorization, Content-Type, etc.
)
# ────────────────────────────────────────────────────────────────────────

# ———————————————————————————————
# 1) Load your model & record its expected feature names
# ———————————————————————————————
@app.on_event("startup")
def load_model():
    global model, expected_features
    try:
        # fix: correct filename
        with open("crop_recommendation_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # LightGBM sklearn wrapper exposes feature_name_
    if hasattr(model, "feature_name_"):
        expected_features = list(model.feature_name_)
    # fallback to raw Booster
    elif hasattr(model, "booster_"):
        expected_features = model.booster_.feature_name()
    else:
        raise RuntimeError("Model does not expose feature names")

    print(f"🔍 Model expects these {len(expected_features)} features:")
    print(expected_features)


# ———————————————————————————————
# 2) Health check
# ———————————————————————————————
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ———————————————————————————————
# 3) Predict endpoint using only those features
# ———————————————————————————————
@app.post("/predict")
async def recommend_crop(payload: dict = Body(..., example={
    # example showing the 7 features your model needs
    "K": 50.0,
    "N": 0.3,
    "P": 35.0,
    "humidity": 78.5,
    "ph": 6.8,
    "rainfall": 120.0,
    "temperature": 22.5
})):
    # Build a one-row DataFrame from the incoming JSON
    df = pd.DataFrame([payload])

    # Check for missing keys
    missing = set(expected_features) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {sorted(missing)}"
        )

    # Slice & reorder exactly as the model expects
    X = df[expected_features].values

    # Predict
    try:
        pred = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"recommended_crop": pred[0]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
