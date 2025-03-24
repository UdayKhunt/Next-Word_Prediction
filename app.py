import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

model = load_model('next_word_lstm.keras')
with open('tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)

def predict_next_word(tokenizer , model , text , max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) > max_seq_len:
        token_list = token_list[-(max_seq_len) : ]
    token_list_padded = pad_sequences([token_list] , maxlen=max_seq_len , padding='pre')
    predicted = model.predict(token_list_padded) #ohe
    predicted_word_index = np.argmax(predicted , axis = 1)
    for word,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(tokenizer, model, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')


