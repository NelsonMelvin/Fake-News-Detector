import pandas as pd
import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load and prepare data
df = pd.read_csv("Fake_small.csv")
df_real = pd.read_csv("True_small.csv")

df['label'] = 0  # Fake
df_real['label'] = 1  # Real

data = pd.concat([df, df_real]).sample(frac=1).reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

data['text'] = data['text'].apply(clean_text)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Paste a news article below to check if it's real or fake.")

user_input = st.text_area("Paste article text here:")

if st.button("Check News"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    result = "🟢 Real" if prediction == 1 else "🔴 Fake"
    st.success(f"This news is likely: {result}")
