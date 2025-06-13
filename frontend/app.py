# Placeholder for app.py
import streamlit as st
import requests

st.title("🧠 Spotting the Fakes!")
BACKEND_URL = "http://127.0.0.1:5000"
mode = st.sidebar.radio("Choose Detection Mode", ["Text", "Image", "Face"])

if mode == "Text":
    text = st.text_area("Enter text")
    if st.button("Check"):
        res = requests.post(f"{BACKEND_URL}/api/text/predict", json={"text": text})
        st.write(res.json())

elif mode == "Image":
    file = st.file_uploader("Upload Image")
    if file and st.button("Analyze"):
        res = requests.post(f"{BACKEND_URL}/api/image/predict", files={"file": file})
        st.write(res.json())

elif mode == "Face":
    file = st.file_uploader("Upload Face Image")
    if file and st.button("Analyze"):
        res = requests.post(f"{BACKEND_URL}/api/face/predict", files={"file": file})
        st.write(res.json())
