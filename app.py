#importing necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import os
import base64


# UI Enhancements
st.set_page_config(page_title="AI Sentence Auto-Complete", page_icon="📝", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #1a1a2e;
        color: #00d4ff;
        border: 1px solid #00d4ff;
        border-radius: 10px;
        padding: 10px;
    }
    .stSlider > div > div > div > div {
        background-color: #00d4ff;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: #0d1117;
        border-radius: 20px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #007bb5;
        color: white;
        transform: scale(1.05);
    }
    .success-box {
        padding: 20px;
        border-radius: 15px;
        background: rgba(0, 212, 255, 0.1);
        border-left: 5px solid #00d4ff;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 AI Sentence Auto-Complete")
st.subheader("Predicting the next words with LSTM Neural Networks")

#loading the pre-trained weights and model architecture
model = tf.keras.models.load_model('MODELS/AUTO_COM_model.h5', custom_objects={'LSTM': LSTM, 'Bidirectional': Bidirectional})

# No GIF needed for modern clean UI
data_url = "" 

#dataset preprocessing
file = open("DATA/data.txt").read() #opeining the dataset and reading from it

tokenizer = Tokenizer() #tokenizing the dataset
data = file.lower().split("\n") #converting dataset to lowercase

#removing whitespaces from the dataset
corpus = []
for line in data:
    a = line.strip()
    corpus.append(a)

#generating tokens for each sentence in the data
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
# print(tokenizer.word_index)
# print(total_words)

#creating labels for each sentence in dataset
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)



#generating next words given a seed
def next_word(seed):
  seed_text = seed
  next_words = num
  for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    import numpy as np
    predicted_probs = model.predict(token_list, verbose=0)
    predicted = np.argmax(predicted_probs, axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
      if index == predicted:
        output_word = word
        break
    seed_text += " " + output_word
  st.markdown(f'<div class="success-box"><h3>Completed Sentence:</h3><p style="font-size: 1.2rem;">{seed_text}</p></div>', unsafe_allow_html=True)

menu = ["AUTO COMPLETE", "ME"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "AUTO COMPLETE":
  st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 50px;'>🤖</h1>
        <p>Your Intelligent Sentence Assistant</p>
    </div>
  """, unsafe_allow_html=True)
  num = st.slider("Number of predictions", 1, 5, 3)
  next_word(st.text_input('Enter your sentence starting phrase', 'The future is'))

else:
  st.header("👤 About the Developer")
  st.write("### Abhishek Maheshwari")
  st.write("AI & Machine Learning Enthusiast")
  st.write("---")
  st.write("📫 **Connect with me:**")
  st.write("[GitHub](https://github.com/Abhishek-Maheshwari-778)")
  st.write("[LinkedIn](https://www.linkedin.com/in/abhishek-maheshwari/)") # Assuming a standard slug
  st.info("This project uses a Deep Learning LSTM model to predict and complete sentences based on learned patterns.")

