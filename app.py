import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import requests
import time
import json

# ==============================================================================
# Helper Functions
# ==============================================================================

# --- Function to get the full extract from Wikipedia API ---
def get_full_style_info_from_wikipedia(style_name):
    """
    Fetches the entire summary extract for a given art style from the Wikipedia API.
    """
    try:
        S = requests.Session()
        URL = "https://en.wikipedia.org/w/api.php"
        PARAMS = {
            "action": "query", "prop": "extracts", "exintro": False, "explaintext": True,
            "format": "json", "titles": style_name
        }
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        PAGES = DATA["query"]["pages"]
        for k, v in PAGES.items():
            # Return the full extract, not just the first paragraph
            return v.get("extract", "No summary available.")
    except Exception as e:
        return f"Could not fetch details from Wikipedia: {e}"

# --- Function to generate a summary and artist list using Gemini (structured JSON output) ---
def generate_summary_and_artists(style, wikipedia_extract):
    """
    Uses Gemini to summarize the Wikipedia text and extract a list of artists,
    returning the result as a structured JSON object.
    """
    max_retries = 5
    delay = 1
    for i in range(max_retries):
        try:
            prompt = (
                f"Based on the following text about {style}, please perform two tasks:\n"
                f"1. Write a concise, one-paragraph summary.\n"
                f"2. Extract a list of up to 5 of the most influential artists mentioned.\n\n"
                f"Text:\n\"\"\"\n{wikipedia_extract}\n\"\"\""
            )

            generation_config = {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "summary": {"type": "STRING"},
                        "artists": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        }
                    }
                }
            }
            
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": generation_config
            }
            # UPDATED: Use st.secrets to get the API key
            api_key = st.secrets["GEMINI_API_KEY"]
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
            
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
            response.raise_for_status()
            
            result = response.json()

            if (result.get('candidates') and result['candidates'][0].get('content') and 
                result['candidates'][0]['content'].get('parts')):
                # The response part is a JSON string, so we need to parse it
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                return json.loads(json_string)
            else:
                return {"summary": "Could not generate summary.", "artists": []}

        except requests.exceptions.RequestException as e:
            if i < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"Failed to connect to the AI model for summary/artist generation: {e}")
                return {"summary": "Error connecting to AI model.", "artists": []}
    return {"summary": "Could not generate summary after retries.", "artists": []}


# --- Function to call the Gemini API for the final explanation ---
def generate_ai_explanation(style, summary, artists):
    """
    Uses the AI-generated summary and artist list to create the final explanation.
    """
    max_retries = 5
    delay = 1
    for i in range(max_retries):
        try:
            artist_list = ", ".join(artists) if artists else "various artists"
            prompt = (
                f"You are an engaging and knowledgeable museum tour guide. "
                f"A visitor is looking at a piece of {style} art. "
                f"Explain what {style} is in a short, accessible, and interesting paragraph. "
                f"Use these facts to help you:\n\n"
                f"- Definition: \"{summary}\"\n"
                f"- Key Artists: \"{artist_list}\"\n\n"
                f"Do not just repeat the facts; weave them into a compelling narrative."
            )

            chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
            payload = {"contents": chat_history}
            # UPDATED: Use st.secrets to get the API key
            api_key = st.secrets["GEMINI_API_KEY"]
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
            
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
            response.raise_for_status()
            
            result = response.json()
            
            if (result.get('candidates') and result['candidates'][0].get('content') and 
                result['candidates'][0]['content'].get('parts')):
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Could not generate an explanation at this time."

        except requests.exceptions.RequestException as e:
            if i < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"Failed to connect to the AI model for the final explanation: {e}")
                return "Error: Could not connect to the AI model."
    return "Could not generate an explanation after several retries."


# ==============================================================================
# Load Models and Pre-processing Objects
# ==============================================================================

@st.cache_resource
def load_models():
    """
    Loads all the necessary model files once.
    """
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        IMG_SIZE = (224, 224)
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(
            weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,)
        )
        base_model.trainable = False
        pooled_output = GlobalAveragePooling2D()(base_model.output)
        feature_extractor = Model(inputs=base_model.input, outputs=pooled_output)
        
        return model, scaler, encoder, feature_extractor
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Make sure 'best_model.pkl', 'scaler.pkl', and 'label_encoder.pkl' are in the same directory.")
        return None, None, None, None

ml_model, scaler, le, feature_extractor = load_models()

# ==============================================================================
# Streamlit App UI
# ==============================================================================

st.set_page_config(layout="wide")
st.title("🎨 AI Art Advisor")
st.write("Upload a picture of a painting to classify its art style and get an AI-generated explanation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and all([ml_model, scaler, le, feature_extractor]):
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Artwork', use_column_width=True)

    with col2:
        with st.spinner('Analyzing the artwork...'):
            # 1. Preprocess and predict style
            image_resized = image.resize((224, 224))
            image_array = np.array(image_resized)
            image_expanded = np.expand_dims(image_array, axis=0)
            image_preprocessed = preprocess_input(image_expanded)
            features = feature_extractor.predict(image_preprocessed)
            features_scaled = scaler.transform(features)
            prediction_encoded = ml_model.predict(features_scaled)[0]
            predicted_style = le.inverse_transform([prediction_encoded])[0]

            st.success(f"**Predicted Art Style:** {predicted_style}")

        with st.spinner('Fetching details and generating explanation...'):
            # 2. Get full text from Wikipedia
            full_extract = get_full_style_info_from_wikipedia(predicted_style)
            
            if "Could not fetch" not in full_extract:
                # 3. Use Gemini to summarize text and extract artists
                ai_generated_data = generate_summary_and_artists(predicted_style, full_extract)
                ai_summary = ai_generated_data.get("summary", "Summary could not be generated.")
                ai_artists = ai_generated_data.get("artists", [])
                
                # 4. Use Gemini again to generate the final explanation
                final_explanation = generate_ai_explanation(predicted_style, ai_summary, ai_artists)
                
                st.subheader("About the Style")
                st.write(final_explanation)
            else:
                st.error(full_extract)

elif uploaded_file is not None:
    st.error("Model files could not be loaded. Please check the console for errors.")
