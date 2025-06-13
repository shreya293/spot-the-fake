# Placeholder for video_service.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../AIML/video-model/model/video_cnn_model.h5"))
model = load_model(MODEL_PATH)
IMG_SIZE = (128, 128)

def predict_frame(frame):
    frame = cv2.resize(frame, IMG_SIZE)
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=0)
    pred = model.predict(frame)[0]
    label = "real" if np.argmax(pred) == 0 else "fake"
    confidence = float(np.max(pred))
    return label, confidence

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count, real, fake = 0, 0, 0
    frame_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        label, conf = predict_frame(frame)
        frame_results.append({"label": label, "confidence": round(conf, 2)})
        if label == "real":
            real += 1
        else:
            fake += 1
        frame_count += 1

    cap.release()

    return {
        "summary": {
            "total": frame_count,
            "real": real,
            "fake": fake
        },
        "frames": frame_results
    }
