# Placeholder for train.py
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "Breaking news! The president resigns today.",
    "Earn $10000 per day by working from home!",
    "NASA discovers new planet with signs of life.",
    "This miracle drug can cure all diseases!",
]
labels = [0, 1, 0, 1]  # 0 = real, 1 = fake

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("✅ Text model saved.")
