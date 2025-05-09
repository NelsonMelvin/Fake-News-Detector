import joblib
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")  # Load the model using joblib
    vectorizer = joblib.load("vectorizer.pkl")  # Load the vectorizer using joblib
    return model, vectorizer

model, vectorizer = load_model()

st.title("Fake News Detector")
text_input = st.text_area("Paste a news article:")

import os
import pickle

# Get the current directory of app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer using absolute paths
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))


if st.button("Check if it's Fake or Real"):
    cleaned = clean_text(text_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    result = "Real" if prediction == 1 else "Fake"
    st.write("This news is:", result)
