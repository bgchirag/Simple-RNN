# Step 1: Import Libraries and Load the Model
import streamlit as st

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from pathlib import Path
import json

MODEL_PATH = Path(__file__).resolve().parent / "simple_rnn_imdb.h5"
WORD_INDEX_PATH = Path(__file__).resolve().parent / "artifacts" / "imdb_word_index.json"
WORD_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_model_cached():
    with st.spinner('Loading model...'):
        model = load_model(MODEL_PATH)
        model.predict(np.zeros((1, 500), dtype=np.int32), verbose=0)
        return model

@st.cache_data(show_spinner=False)
def load_word_index():
    if WORD_INDEX_PATH.exists():
        with WORD_INDEX_PATH.open('r', encoding='utf-8') as f:
            word_index = json.load(f)
    else:
        with st.spinner('Downloading word index...'):
            word_index = imdb.get_word_index()
        with WORD_INDEX_PATH.open('w', encoding='utf-8') as f:
            json.dump(word_index, f)
    word_index = {key: int(value) for key, value in word_index.items()}
    reverse_word_index = {int(value): key for key, value in word_index.items()}
    return word_index, reverse_word_index

word_index, reverse_word_index = load_word_index()
model = load_model_cached()

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


## streamlit app
# Streamlit app
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if not user_input.strip():
        st.warning('Please enter a movie review.')
    else:
        with st.spinner('Preprocessing text...'):
            preprocessed_input = preprocess_text(user_input)

        with st.spinner('Analyzing sentiment...'):
            prediction = model.predict(preprocessed_input, verbose=0)
        
        probability = float(prediction[0][0])
        sentiment = 'Positive' if probability > 0.5 else 'Negative'

        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Confidence Score:** {probability:.4f}')
        
        if probability > 0.7:
            st.success('Strong positive sentiment!')
        elif probability < 0.3:
            st.error('Strong negative sentiment!')
        else:
            st.warning('Neutral sentiment')
else:
    st.write('Please enter a movie review.')


