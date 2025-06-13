# Placeholder for train.py
# AIML/video_model/train.py
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = (128, 128)
X, y = [], []

def extract_frames(video_path, label, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame = frame.astype("float32") / 255.0
        X.append(frame)
        y.append(label)
        count += 1
    cap.release()

# Load data from 'data/real' and 'data/fake'
for folder, label in [("real", 0), ("fake", 1)]:
    folder_path = os.path.join("data", folder)
    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            extract_frames(os.path.join(folder_path, file), label)

X = np.array(X)
y = to_categorical(y, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

os.makedirs("model", exist_ok=True)
model.save("model/video_cnn_model.h5")
print("✅ Video model trained and saved to model/video_cnn_model.h5")
