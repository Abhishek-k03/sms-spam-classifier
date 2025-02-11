import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import os

# Download necessary NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Stemmer
ps = PorterStemmer()

# Set Page Configuration
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# Custom CSS for Styling
st.markdown("""
    <style>
        body {
            background-color: #F3F4F6;
        }
        .main {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        .stTextInput, .stTextArea {
            border-radius: 10px;
        }
        .spam {
            color: red;
            font-size: 24px;
            font-weight: bold;
        }
        .not-spam {
            color: green;
            font-size: 24px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Load vectorizer & trained model
if os.path.exists("vectorizer.pkl") and os.path.exists("model.pkl"):
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
else:
    st.error("❌ Error: Model or vectorizer file not found! Make sure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
    st.stop()

# Header Section
st.markdown("<h1 style='text-align: center;'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Enter a message below to check if it's Spam or Not.</p>", unsafe_allow_html=True)

# Input field
input_sms = st.text_area("✉️ Enter the message:", height=120)

# Text preprocessing function
def transform_text(text):
    if not text.strip():
        return ""  # Handle empty input

    text = text.lower()  # Lowercase
    text = word_tokenize(text)  # Tokenization

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words("english")]

    # Apply stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# Run classification only if user entered text
if st.button("🔍 Analyze Message"):
    if input_sms:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Check if transformation is valid
        if transformed_sms:
            # Vectorize the input
            vector_input = tfidf.transform([transformed_sms])

            # Predict
            result = model.predict(vector_input)[0]

            # Display result
            if result == 1:
                st.markdown("<p class='spam' style='text-align: center; font-size:32px;'>🚨 SPAM 🚨</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='not-spam' style='text-align: center;'>✅ NOT SPAM ✅</p>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ The entered message contains only stopwords or empty text.")
    else:
        st.warning("⚠️ Please enter a message before analyzing.")
