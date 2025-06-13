# Placeholder for train.py
# AIML/face-model/train.py
import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

IMG_SIZE = (128, 128)

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipped unreadable image: {image_path}")
        return None
    img = cv2.resize(img, IMG_SIZE)
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

def load_data(data_dir):
    X, y = [], []
    for label in ["Real", "Fake"]:
        folder_path = os.path.join(data_dir, label)
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue
        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder_path, file)
                features = extract_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

# Load dataset
X, y = load_data("data/face_dataset")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and encoder
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/face_cnn_model.pkl")
joblib.dump(le, "model/face_label_encoder.pkl")

print("✅ Model and label encoder saved.")
