# Placeholder for predict.py
import joblib

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_text(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]
    label = "real" if pred == 0 else "fake"
    return label, round(prob, 2)
