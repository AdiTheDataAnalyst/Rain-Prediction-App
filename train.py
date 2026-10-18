"""Run once (or whenever rain.csv changes):  python train.py"""
import json
from model_utils import save_bundle

bundle = save_bundle()
print(json.dumps(bundle["metrics"], indent=2))
print("Saved rain_model.pkl")
