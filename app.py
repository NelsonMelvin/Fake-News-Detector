import streamlit as st
import pandas as pd
import string
import joblib
import nltk
from nltk.corpus import stopwords

# Download stopwords
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
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"Failed to load model or vectorizer: {e}")
    st.stop()

# Streamlit UI
st.title("📰 Fake News Detector")

text_input = st.text_area("Paste a news article:")

if st.button("Check if it's Fake or Real"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(text_input)
        try:
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            result = "Real" if prediction == 1 else "Fake"
            st.success(f"This news is: **{result}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
