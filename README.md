# spot-the-fake 

This project is an AI-based detection tool for determining whether the input is real or fake, using machine learning and deep learning models.

It is able to detect all four formats:

📰 Text (News/Headlines)

🖼️ Images

🧑‍🦰 Face Photos

🎥 Video Deepfakes

🔍 Key Features

Detect fake news with TF-IDF + Logistic Regression.

Detect tampered or fake images using color histograms.

Detect AI-generated (GAN) faces using CNN.

Analyze videos frame by frame to detect fake faces.

💻 Technogies Used

Python, Flask, Streamlit

Scikit-learn, TensorFlow/Keras, OpenCV

Model formats: .pkl for classical ML, .h5 for CNN models.

🚀 How to Run

Run Flask backend:

cd backend python app.py 2.Rum streamlit frontend:

cd ../frontend streamlit run app.py


cd backend python app.py 2.Rum streamlit frontend:

cd ../frontend streamlit run app.py
