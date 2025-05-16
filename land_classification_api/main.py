import os
import torch
import torchvision.models as models
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from get_sentinel_tile import get_sentinel_tile
import base64
from io import BytesIO
import pickle

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model ---
# Allow unpickling of ResNet
torch.serialization.add_safe_globals([models.resnet.ResNet])
state_path = os.path.join(os.path.dirname(__file__), "land_classifier.pt")
model_obj = torch.load(state_path, map_location="cpu", weights_only=False)
if isinstance(model_obj, dict):
    model = models.resnet50(pretrained=False)
    model.load_state_dict(model_obj)
else:
    model = model_obj
model.eval()

# --- Load artifacts ---
base_dir = os.path.dirname(__file__)
with open(os.path.join(base_dir, "class_names.pkl"), "rb") as f:
    class_names = pickle.load(f)
with open(os.path.join(base_dir, "transform.pkl"), "rb") as f:
    transform = pickle.load(f)

@app.get("/predict")
def predict(lat: float = Query(...), lon: float = Query(...)):
    # 1) fetch Sentinel tile
    image = get_sentinel_tile(lat, lon)

    # 2) encode tile as base64
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 3) preprocess and infer
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    pred_idx = output.argmax(dim=1).item()
    label = class_names[pred_idx]

    # 4) return both prediction and tile
    return {"prediction": label, "tile": img_b64}