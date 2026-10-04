import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load tokenizer
# -----------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -----------------------------
# Load max sequence length
# -----------------------------
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("lstm_model.h5")

# -----------------------------
# Function to predict next word
# -----------------------------
def predict_next_word(text):

    # Convert text to sequence
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Pad sequence
    token_list = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding='pre'
    )

    # Predict probabilities
    predicted = model.predict(token_list, verbose=0)

    # Get highest probability index
    predicted_word_index = np.argmax(predicted, axis=-1)[0]

    # Convert index to word
    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            output_word = word
            break

    return output_word

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 LSTM Next Word Prediction")
st.write("Type a sentence and predict the next word.")

# User input
input_text = st.text_input(
    "Enter text",
    placeholder="Enter a sentence..."
)

# Predict button
if st.button("Predict Next Word"):

    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(input_text)

        st.success(f"Predicted Next Word: **{next_word}**")