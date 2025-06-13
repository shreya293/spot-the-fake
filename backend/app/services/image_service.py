# Placeholder for image_service.py
import cv2
import numpy as np
import joblib
import os

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../AIML/image_model/model/model.pkl"))
model = joblib.load(MODEL_PATH)

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

def predict_image_label(image_path):
    features = extract_features(image_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    confidence = round(model.predict_proba(features)[0][prediction], 2)
    label = "real" if prediction == 0 else "fake"
    return label, confidence
