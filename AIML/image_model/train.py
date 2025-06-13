# Placeholder for train.py
import os
import cv2
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Skipped: {image_path}")
        return None
    image = cv2.resize(image, (128, 128))
    hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

def load_data(folder_path):
    X, y = [], []
    for label, folder in enumerate(["Real", "Fake"]):
        class_path = os.path.join(folder_path, folder)
        if not os.path.exists(class_path):
            print(f"❌ Missing: {class_path}")
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                feat = extract_features(os.path.join(class_path, file))
                if feat is not None:
                    X.append(feat)
                    y.append(label)
    return np.array(X), np.array(y)

X, y = load_data("data/validation")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("✅ Saved: model/model.pkl")
