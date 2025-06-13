# Placeholder for text_service.py
import os
import joblib
import numpy as np

# Load model, vectorizer, label encoder
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../AIML/text_model/model"))
model = joblib.load(os.path.join(MODEL_DIR, "text_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

def predict_text_label(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    proba = model.predict_proba(text_vector)
    label = label_encoder.inverse_transform(prediction)[0]
    confidence = round(np.max(proba), 2)
    return label, confidence
