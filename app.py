import os
import pickle
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

# Define a cache function to load the model and vectorizer
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "model.pkl")  # Use current working directory
    vectorizer_path = os.path.join(os.getcwd(), "vectorizer.pkl")  # Use current working directory

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

# Load model and vectorizer using the cache function
model, vectorizer = load_model()

st.title("Fake News Detector")
text_input = st.text_area("Paste a news article:")

if st.button("Check if it's Fake or Real"):
    cleaned = clean_text(text_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    result = "Real" if prediction == 1 else "Fake"
    st.write("This news is:", result)
