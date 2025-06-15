import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM, Add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import re
from collections import Counter

# Configuration
MAX_LENGTH = 34
MODEL_PATH = 'models/best_model.h5'
PREPROCESSED_PATH = 'preprocessed'

# Page setup
st.set_page_config(page_title="Image Captioning Demo", layout="wide")
st.title("📷 Image Captioning System")
st.markdown("""
This application generates captions for images using either:
1. Our custom-trained model (LSTM-based)
2. BLIP-2 (state-of-the-art model from Salesforce)
""")

# Sidebar
st.sidebar.header("Settings")
model_choice = st.sidebar.radio(
    "Select captioning model:",
    ("Custom Trained Model", "BLIP-2 (Advanced)")
)
beam_width = st.sidebar.slider(
    "Beam width (for custom model)", 
    min_value=1, max_value=10, value=3
)
repetition_penalty = st.sidebar.slider(
    "Repetition penalty", 
    min_value=1.0, max_value=2.0, value=1.2, step=0.1
)

# Load resources
@st.cache_resource
def load_resources():
    """Load tokenizer, captions, and model"""
    try:
        # Load tokenizer and captions
        with open(os.path.join(PREPROCESSED_PATH, 'tokenizer.pkl'), 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open(os.path.join(PREPROCESSED_PATH, 'captions.pkl'), 'rb') as f:
            captions = pickle.load(f)
        
        # Load model
        def create_model():
            inputs1 = Input(shape=(2048,))
            fe1 = Dropout(0.5)(inputs1)
            fe2 = Dense(512, activation='relu')(fe1)

            inputs2 = Input(shape=(MAX_LENGTH,))
            se1 = Embedding(len(tokenizer.word_index) + 1, 256, mask_zero=True)(inputs2)
            se2 = Dropout(0.5)(se1)
            se3 = LSTM(512)(se2)

            decoder1 = Add()([fe2, se3])
            decoder2 = Dense(512, activation='relu')(decoder1)
            outputs = Dense(len(tokenizer.word_index) + 1, activation='softmax')(decoder2)

            model = Model(inputs=[inputs1, inputs2], outputs=outputs)
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
            return model

        model = create_model()
        model.load_weights(MODEL_PATH)
        
        # Load feature extractor
        cnn_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        
        return tokenizer, captions, model, cnn_model
    
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, None, None

tokenizer, captions, model, cnn_model = load_resources()

# Helper functions
def postprocess_caption(caption):
    """Clean up generated caption"""
    caption = re.sub(r'\b(\w+)( \1\b)+', r'\1', caption)  # remove repeated words
    caption = caption.strip().capitalize()
    caption = re.sub(r'[ ]+', ' ', caption)
    caption = re.sub(r'[.?!,;:]*$', '', caption) + '.'  # Ensure ends with period
    return caption

def extract_features(image, cnn_model):
    """Extract image features using InceptionV3"""
    img = image.resize((299, 299))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = cnn_model.predict(img_array)
    return features.reshape(1, 2048)

def generate_caption_custom(image_features, beam_index=3, repetition_penalty=1.2):
    """Generate caption using custom trained model"""
    start = [tokenizer.word_index['startseq']]
    sequences = [[start, 0.0]]

    while len(sequences[0][0]) < min(MAX_LENGTH, 20):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == tokenizer.word_index['endseq']:
                all_candidates.append([seq, score])
                continue
                
            padded = pad_sequences([seq], maxlen=MAX_LENGTH)
            preds = model.predict([image_features, padded], verbose=0)[0]

            # Apply repetition penalty
            word_counts = Counter(seq)
            for word, count in word_counts.items():
                if word < len(preds):
                    preds[word] /= (count ** repetition_penalty)

            top_preds = np.argsort(preds)[-beam_index:]

            for word in top_preds:
                new_seq = seq + [word]
                new_score = score + np.log(preds[word] + 1e-9)
                all_candidates.append([new_seq, new_score])

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_index]

    final_seq = sequences[0][0]
    final_caption = [tokenizer.index_word.get(i, '') for i in final_seq]
    final_caption = ' '.join(final_caption[1:]).split('endseq')[0].strip()
    return postprocess_caption(final_caption)

def generate_caption_blip(image):
    """Generate caption using BLIP-2 model"""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        
        inputs = processor(images=image.convert('RGB'), return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = blip_model.generate(
                **inputs,
                max_length=30,
                num_beams=5,
                early_stopping=True
            )
        
        caption = processor.decode(output[0], skip_special_tokens=True)
        return postprocess_caption(caption)
    
    except Exception as e:
        st.error(f"BLIP-2 Error: {str(e)}")
        return None

# Main interface
uploaded_file = st.file_uploader(
    "Upload an image for captioning", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("Generated Caption")
        
        if model_choice == "Custom Trained Model":
            if model is None:
                st.error("Custom model not loaded properly")
            else:
                with st.spinner("Generating caption (custom model)..."):
                    features = extract_features(image, cnn_model)
                    caption = generate_caption_custom(
                        features, 
                        beam_index=beam_width,
                        repetition_penalty=repetition_penalty
                    )
                    st.success(caption)
        else:
            with st.spinner("Generating caption (BLIP-2)..."):
                caption = generate_caption_blip(image)
                if caption:
                    st.success(caption)
                else:
                    st.error("Failed to generate caption with BLIP-2")

# Sample images section
st.subheader("Try with sample images")
sample_images = {
    "Beach": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
    "Mountains": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b",
    "City": "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df"
}

cols = st.columns(len(sample_images))
for idx, (name, url) in enumerate(sample_images.items()):
    with cols[idx]:
        st.image(url, caption=name, width=200)
        if st.button(f"Caption {name}", key=name):
            try:
                import requests
                from io import BytesIO
                
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                
                st.session_state.sample_image = image
                st.session_state.sample_image_name = name
                st.rerun()
                
            except Exception as e:
                st.error(f"Error loading sample image: {str(e)}")

# Handle sample image selection
if "sample_image" in st.session_state:
    st.subheader(f"Sample Image: {st.session_state.sample_image_name}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(st.session_state.sample_image, use_column_width=True)
    
    with col2:
        st.subheader("Generated Caption")
        
        if model_choice == "Custom Trained Model":
            with st.spinner("Generating caption (custom model)..."):
                features = extract_features(st.session_state.sample_image, cnn_model)
                caption = generate_caption_custom(
                    features,
                    beam_index=beam_width,
                    repetition_penalty=repetition_penalty
                )
                st.success(caption)
        else:
            with st.spinner("Generating caption (BLIP-2)..."):
                caption = generate_caption_blip(st.session_state.sample_image)
                if caption:
                    st.success(caption)
                else:
                    st.error("Failed to generate caption with BLIP-2")