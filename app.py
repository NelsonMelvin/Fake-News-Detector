import streamlit as st
import joblib

# Load the full pipeline
try:
    model = joblib.load("pipeline.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("📰 Fake News Detector")

text_input = st.text_area("Paste a news article:")

if st.button("Check if it's Fake or Real"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        try:
            prediction = model.predict([text_input])[0]
            result = "Real" if prediction == 1 else "Fake"
            st.success(f"This news is: **{result}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
