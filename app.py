import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM as KerasLSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, load_model
import streamlit as st
from PIL import Image
import os
import base64
import numpy as np

# --- Keras 3 Compatibility Layer ---
@tf.keras.utils.register_keras_serializable(package="Custom", name="LSTM")
class CompatibleLSTM(KerasLSTM):
    def __init__(self, **kwargs):
        # Remove unsupported 'time_major' argument from legacy models
        kwargs.pop('time_major', None)
        super().__init__(**kwargs)

# --- UI Configuration ---
st.set_page_config(page_title="AI Sentence Auto-Complete", page_icon="🤖", layout="wide")

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
        padding: 12px;
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
        padding: 25px;
        border-radius: 15px;
        background: rgba(0, 212, 255, 0.1);
        border-left: 5px solid #00d4ff;
        margin-top: 25px;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 AI Sentence Auto-Complete")
st.subheader("Deep Learning (LSTM) Next Word Prediction")

# --- Model Loading ---
@st.cache_resource
def load_ai_model():
    model_path = 'MODELS/AUTO_COM_model.h5'
    try:
        # We map 'LSTM' to our CompatibleLSTM class to handle the legacy config
        return tf.keras.models.load_model(
            model_path, 
            custom_objects={'LSTM': CompatibleLSTM, 'Bidirectional': Bidirectional}
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_ai_model()

# --- Data Preprocessing ---
@st.cache_data
def get_tokenizer():
    if not os.path.exists("DATA/data.txt"):
        return None, 0, 0
    file = open("DATA/data.txt").read()
    tokenizer = Tokenizer()
    data = file.lower().split("\n")
    corpus = [line.strip() for line in data if line.strip()]
    tokenizer.fit_on_texts(corpus)
    
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    
    max_sequence_len = max([len(x) for x in input_sequences]) if input_sequences else 0
    total_words = len(tokenizer.word_index) + 1
    return tokenizer, max_sequence_len, total_words

tokenizer, max_sequence_len, total_words = get_tokenizer()

# --- Prediction Logic ---
def generate_suggestion(seed_text, next_words_count):
    if model is None or tokenizer is None:
        return "Model not loaded properly."
    
    result_text = seed_text
    for _ in range(next_words_count):
        token_list = tokenizer.texts_to_sequences([result_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        result_text += " " + output_word
    return result_text

# --- Sidebar & Navigation ---
menu = ["AUTO COMPLETE", "👤 ABOUT DEVELOPER"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "AUTO COMPLETE":
    st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <h1 style='font-size: 60px;'>🤖</h1>
            <p style='font-size: 1.2rem; opacity: 0.8;'>Type a phrase and let the AI finish your thought.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_input('Start your sentence...', 'I would like to')
    with col2:
        num_preds = st.slider("Words to predict", 1, 10, 3)

    if st.button("Complete Sentence"):
        with st.spinner('Thinking...'):
            final_text = generate_suggestion(user_input, num_preds)
            st.markdown(f'''
                <div class="success-box">
                    <h3>🔮 Prediction:</h3>
                    <p style="font-size: 1.4rem; color: #00d4ff;">"{final_text}"</p>
                </div>
            ''', unsafe_allow_html=True)

else:
    st.header("👤 About the Developer")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown("<h1 style='font-size: 100px;'>👨‍💻</h1>", unsafe_allow_html=True)
    with col_b:
        st.write("### Abhishek Maheshwari")
        st.write("AI & Machine Learning Enthusiast")
        st.write("Specializing in Natural Language Processing and Neural Networks.")
        st.write("---")
        st.write("📫 **Connect with me:**")
        st.write("[GitHub](https://github.com/Abhishek-Maheshwari-778)")
        st.write("[LinkedIn](https://www.linkedin.com/in/abhishek-maheshwari/)")
    
    st.info("This system uses a Bidirectional LSTM (Long Short-Term Memory) architecture to understand context and predict successive words.")
