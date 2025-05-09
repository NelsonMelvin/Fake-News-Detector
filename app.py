import streamlit as st
import pandas as pd
import string
import joblib
import nltk
from nltk.corpus import stopwords

# Download stopwords (only runs once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean the input text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detector")

text_input = st.text_area("Paste a news article:")

if st.button("Check if it's Fake or Real"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(text_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        result = "Real" if prediction == 1 else "Fake"
        st.success(f"This news is: **{result}**")
