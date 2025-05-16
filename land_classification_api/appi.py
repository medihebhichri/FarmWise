# inspect_model.py
import pickle

with open("crop_recommendation_model.pkl","rb") as f:
    obj = pickle.load(f)

print("Type:", type(obj))
print("Has predict? ", hasattr(obj, "predict"))
