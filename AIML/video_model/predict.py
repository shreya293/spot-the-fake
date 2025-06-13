# Placeholder for predict.py
# AIML/video_model/predict.py
import os
import cv2
import numpy as np
from keras.models import load_model

model = load_model("model/video_cnn_model.h5")
IMG_SIZE = (128, 128)

def predict_frame(frame):
    frame = cv2.resize(frame, IMG_SIZE)
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=0)
    pred = model.predict(frame)[0]
    label = "real" if np.argmax(pred) == 0 else "fake"
    confidence = float(np.max(pred))
    return label, confidence

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total = 0
    real_count = 0
    fake_count = 0
    frame_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        label, conf = predict_frame(frame)
        frame_results.append({"label": label, "confidence": round(conf, 2)})
        if label == "real":
            real_count += 1
        else:
            fake_count += 1
    cap.release()

    summary = {
        "total": total,
        "real": real_count,
        "fake": fake_count
    }
    return summary, frame_results

# ✅ Example usage
if __name__ == "__main__":
    summary, results = analyze_video("sample.mp4")  # Replace with your video
    print("Video Summary:", summary)
    print("First 5 frame results:", results[:5])
