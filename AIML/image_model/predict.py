# Placeholder for predict.py
import cv2, numpy as np, joblib

model = joblib.load("model/model.pkl")

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    hist = cv2.calcHist([img], [0,1,2], None, [8]*3, [0,256]*3)
    features = cv2.normalize(hist, hist).flatten().reshape(1, -1)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][pred]
    return "real" if pred == 0 else "fake", round(prob, 2)
