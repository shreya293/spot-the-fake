# Placeholder for face_service.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../AIML/face-model/model/face_cnn_model.h5"))
model = load_model(MODEL_PATH)

def preprocess_face_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict_face_label(image_path):
    img = preprocess_face_image(image_path)
    prediction = model.predict(img)[0]
    label = "real" if np.argmax(prediction) == 0 else "fake"
    confidence = float(np.max(prediction))
    return label, round(confidence, 2)
