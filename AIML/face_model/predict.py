# Placeholder for predict.py
# AIML/face-model/predict.py
import os
import cv2
import numpy as np
import joblib

IMG_SIZE = (128, 128)

# Load model and encoder
model_path = os.path.join("model", "face_cnn_model.pkl")
encoder_path = os.path.join("model", "face_label_encoder.pkl")

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not readable: {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

def predict_face(image_path):
    features = extract_features(image_path).reshape(1, -1)
    pred = model.predict(features)[0]
    conf = model.predict_proba(features).max()
    label = label_encoder.inverse_transform([pred])[0]
    return label, round(conf, 2)

# ✅ Example usage
if __name__ == "__main__":
    img_path = "sample.jpg"  # replace with any test image
    label, conf = predict_face(img_path)
    print(f"Prediction: {label} (Confidence: {conf})")
