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
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt

# Configuration
MAX_LENGTH = 34
MODEL_PATH = 'models/best_model.h5'
PREPROCESSED_PATH = 'preprocessed'
BEAM_INDEX = 10
REPETITION_PENALTY = 1.5

# Initialize global variables
tokenizer = None
model = None
cnn_model = None

# Page setup
st.set_page_config(page_title="AI Image Captioning", layout="wide")
st.title("📷 Advanced Image Captioning System")

def load_resources():
    """Load all required resources with proper error handling"""
    global tokenizer, model, cnn_model
    
    try:
        # Load tokenizer
        tokenizer_path = os.path.join(PREPROCESSED_PATH, 'tokenizer.pkl')
        if not os.path.exists(tokenizer_path):
            st.error(f"Tokenizer file not found at {tokenizer_path}")
            return False
            
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
            
        # Verify required tokens exist
        required_tokens = ['startseq', 'endseq']
        for token in required_tokens:
            if token not in tokenizer.word_index:
                st.error(f"Missing required token in tokenizer: '{token}'")
                return False

        # Load model architecture
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

            return Model(inputs=[inputs1, inputs2], outputs=outputs)

        model = create_model()
        
        # Load model weights
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model weights not found at {MODEL_PATH}")
            return False
        model.load_weights(MODEL_PATH)
        
        # Load feature extractor
        cnn_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        
        return True
        
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return False

def postprocess_caption(caption, style="default"):
    """Clean and format generated caption with style options"""
    if not caption:
        return ""
        
    # Basic cleaning
    caption = re.sub(r'\b(\w+)( \1\b)+', r'\1', caption)  # Remove repeated words
    caption = caption.strip().capitalize()
    caption = re.sub(r'[ ]+', ' ', caption)
    
    # Style-specific processing
    if style == "short":
        # Keep only first 10 words for short & simple version
        words = caption.split()[:10]
        caption = ' '.join(words)
    elif style == "moderate":
        # Keep 15-20 words for moderate version
        words = caption.split()[:20]
        caption = ' '.join(words)
    
    # Ensure caption ends with punctuation
    if not caption.endswith(('.', '!', '?')):
        caption += '.'
        
    return caption

def extract_features(image):
    """Extract image features using InceptionV3"""
    try:
        img = image.resize((299, 299))
        img_array = np.array(img)
        
        # Convert grayscale to RGB if needed
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
            
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = cnn_model.predict(img_array, verbose=0)
        return features.reshape(1, 2048)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def generate_caption_custom(image_features, beam_index=10, repetition_penalty=1.5, style="default"):
    """Generate caption using custom trained model"""
    if tokenizer is None or model is None:
        st.error("Models not loaded properly")
        return ""
        
    try:
        start_token = tokenizer.word_index['startseq']
        sequences = [[[start_token], 0.0]]

        for _ in range(MAX_LENGTH):
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

        # Process final caption
        final_seq = sequences[0][0]
        caption_words = []
        for word_id in final_seq[1:]:  # Skip startseq
            word = tokenizer.index_word.get(word_id, '')
            if word == 'endseq':
                break
            caption_words.append(word)
            
        return postprocess_caption(' '.join(caption_words), style=style)
        
    except Exception as e:
        st.error(f"Caption generation error: {str(e)}")
        return ""

@st.cache_resource
def load_blip_model():
    """Load BLIP-2 model with caching"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        return processor, model, device
    except Exception as e:
        st.error(f"BLIP-2 Loading Error: {str(e)}")
        return None, None, None

def generate_caption_blip(image, style="default"):
    """Generate caption using BLIP-2 model"""
    try:
        processor, blip_model, device = load_blip_model()
        if None in [processor, blip_model, device]:
            return None
            
        inputs = processor(images=image.convert('RGB'), return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = blip_model.generate(
                **inputs,
                max_length=30,
                num_beams=5,
                early_stopping=True
            )
        
        caption = processor.decode(output[0], skip_special_tokens=True)
        return postprocess_caption(caption, style=style)
    except Exception as e:
        st.error(f"BLIP-2 Generation Error: {str(e)}")
        return None

def main():
    st.markdown("Upload an image to generate descriptive captions using either our custom model or BLIP-2")
    
    # Load resources first
    if not load_resources():
        st.error("Failed to load required resources. Please check:")
        st.error(f"- {PREPROCESSED_PATH}/tokenizer.pkl exists")
        st.error(f"- {PREPROCESSED_PATH}/captions.pkl exists")
        st.error(f"- {MODEL_PATH} exists")
        return

    # Sidebar settings
    with st.sidebar:
        st.header("Model Settings")
        model_choice = st.radio(
            "Select Model:",
            ("Custom Trained Model", "BLIP-2 (Advanced)"),
            index=0
        )
        
        st.subheader("Caption Style")
        caption_style = st.radio(
            "Choose caption style:",
            ("Default", "Short & Simple (10 words)", "Moderate (20 words)"),
            index=0
        )
        
        if model_choice == "Custom Trained Model":
            st.subheader("Generation Parameters")
            beam_width = st.slider(
                "Beam width", 
                min_value=1, max_value=15, 
                value=BEAM_INDEX
            )
            repetition_penalty = st.slider(
                "Repetition penalty", 
                min_value=1.0, max_value=3.0, 
                value=REPETITION_PENALTY,
                step=0.1
            )

    # Main interface
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                return

        with col2:
            st.subheader("Generated Caption")
            
            # Map style selection to style parameter
            style_map = {
                "Default": "default",
                "Short & Simple (10 words)": "short",
                "Moderate (20 words)": "moderate"
            }
            style = style_map[caption_style]
            
            if model_choice == "Custom Trained Model":
                with st.spinner("Generating caption..."):
                    features = extract_features(image)
                    if features is not None:
                        caption = generate_caption_custom(
                            features,
                            beam_index=beam_width,
                            repetition_penalty=repetition_penalty,
                            style=style
                        )
                        st.success(caption)
            else:
                with st.spinner("Generating caption with BLIP-2..."):
                    caption = generate_caption_blip(image, style=style)
                    if caption:
                        st.success(caption)
                    else:
                        st.error("Failed to generate caption")
            
            # Display with matplotlib
            if caption:
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(image)
                    ax.axis('off')
                    ax.set_title(f"Generated Caption ({caption_style}):\n{caption}", pad=20)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")

if __name__ == "__main__":
    main()