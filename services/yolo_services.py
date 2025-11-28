from ultralytics import YOLO
import os

MODEL_PATH = "D:\\ui-detector-D2D\\best.pt"  # Path to your trained YOLO model

# Load the model once
model = YOLO(MODEL_PATH)

def run_yolo(image_path):
    results = model.predict(source=image_path, conf=0.5, save=True, verbose=False)
    return results
