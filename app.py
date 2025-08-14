# Import necessary libraries
import gradio as gr
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB7 # Corrected model from previous step
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
import requests
import json
import os
import base64
import io

# --- Define Helper Functions ---

# Helper to convert PIL Image to base64 for Vision API
def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# New: Use Gemini Vision to identify the specific artwork
def identify_artwork_with_gemini(pil_image):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"title": "API key not found.", "artist": None}

        image_b64 = pil_to_base64(pil_image)

        prompt = (
            "Analyze this image of a painting. If it is a well-known artwork, identify its title and artist. "
            "If the artwork is not recognized or is not famous, respond with 'Unknown'. "
            "Provide the output in JSON format with two keys: 'title' and 'artist'."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_b64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

        response = requests.post(api_url, json=payload)
        response.raise_for_status()

        result_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        # Clean potential markdown formatting from the JSON response
        clean_json_text = result_text.strip().replace('```json', '').replace('```', '').strip()
        result_json = json.loads(clean_json_text)
        return result_json

    except Exception as e:
        return {"title": f"Artwork identification error: {e}", "artist": None}


# Get text from Wikipedia
def get_full_style_info_from_wikipedia(style_name):
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
            return v.get("extract", "No summary available.")
    except Exception:
        return "Could not fetch details from Wikipedia."

# Use Gemini to get a summary and list of artists for the STYLE
def generate_summary_and_artists(style, wikipedia_extract):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"summary": "API key not found.", "artists": []}

        prompt = (f"Based on the text about the art style '{style}', write a concise, one-paragraph summary "
                  f"and extract a list of up to 5 influential artists.\n\nText:\n{wikipedia_extract}")
        
        generation_config = {"responseMimeType": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": generation_config}
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
        
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        
        result_json = response.json()['candidates'][0]['content']['parts'][0]['text']
        return json.loads(result_json)

    except Exception as e:
        return {"summary": f"Error generating summary: {e}", "artists": []}

# Use Gemini for the final, user-facing explanation of the STYLE
def generate_ai_explanation(style, summary, artists):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Could not generate explanation: API key not found."

        artist_list = ", ".join(artists) if artists else "various artists"
        prompt = (f"You are a knowledgeable and engaging museum tour guide. Explain what **{style}** art is in an enthusiastic, "
                  f"one-paragraph narrative. Weave in these key facts:\n\n- Definition: \"{summary}\"\n"
                  f"- Key Artists: \"{artist_list}\"")

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
        
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']

    except Exception as e:
        return f"Error generating final explanation: {e}"


# --- Define Model Loading ---

# Load all models from disk
def load_models():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        # Set up the model's input shape for 3-channel RGB
        IMG_SIZE = (224, 224)
        input_tensor = Input(shape=IMG_SIZE + (3,))
        
        # Load EfficientNetB7 to extract image features
        base_model = EfficientNetB7(
            weights='imagenet', include_top=False, input_tensor=input_tensor
        )
        base_model.trainable = False
        
        pooled_output = GlobalAveragePooling2D()(base_model.output)
        feature_extractor = Model(inputs=base_model.input, outputs=pooled_output)
        
        return model, scaler, encoder, feature_extractor
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not load model files: {e}") from e

# Load models and create cache on startup
ml_model, scaler, le, feature_extractor = load_models()
explanation_cache = {}


# --- Define Main Prediction Function ---

# Run this function for each uploaded image
def predict_art_style(input_image):
    # --- New: Artwork Identification ---
    artwork_identity = identify_artwork_with_gemini(input_image)
    artwork_title = artwork_identity.get("title", "Identification failed")
    artwork_artist = artwork_identity.get("artist")

    # --- Existing: Style Classification ---
    image_rgb = input_image.convert("RGB")
    image_resized = image_rgb.resize((224, 224))
    image_array = np.array(image_resized)
    image_expanded = np.expand_dims(image_array, axis=0)
    image_preprocessed = preprocess_input(image_expanded)
    
    features = feature_extractor.predict(image_preprocessed)
    features_scaled = scaler.transform(features)
    prediction_encoded = ml_model.predict(features_scaled)
    predicted_style = le.inverse_transform(prediction_encoded)[0]
    
    label_output = {predicted_style: 1.0}

    # --- Get Style Explanation (with caching) ---
    if predicted_style in explanation_cache:
        style_explanation = explanation_cache[predicted_style]
    else:
        full_extract = get_full_style_info_from_wikipedia(predicted_style)
        if "Could not fetch" in full_extract or "No summary available" in full_extract:
            style_explanation = full_extract
        else:
            ai_data = generate_summary_and_artists(predicted_style, full_extract)
            ai_summary = ai_data.get("summary")
            ai_artists = ai_data.get("artists")
            
            if "Error" in ai_summary:
                style_explanation = ai_summary
            else:
                style_explanation = generate_ai_explanation(predicted_style, ai_summary, ai_artists)
        
        explanation_cache[predicted_style] = style_explanation
    
    # --- New: Combine Outputs for Markdown Display ---
    if artwork_artist and "Unknown" not in artwork_title and "Error" not in artwork_title:
        identification_header = f"### {artwork_title} by {artwork_artist}\n\n"
    elif "Unknown" not in artwork_title and "Error" not in artwork_title:
        identification_header = f"### {artwork_title}\n\n"
    else:
        identification_header = "### Artwork Not Recognized\n\n---\n\n"

    # Combine the identification and the style explanation
    combined_output = identification_header + style_explanation
    
    # Return the final results for display
    return label_output, combined_output

# --- Build Gradio Interface ---

# Build the web app interface
demo = gr.Interface(
    fn=predict_art_style,
    inputs=gr.Image(type="pil", label="Upload an Artwork"),
    outputs=[
        gr.Label(label="Predicted Style"),
        gr.Markdown(label="AI Art Advisor's Analysis") # Updated label
    ],
    title="AI Art Advisor",
    description="Upload a painting to identify the work, classify its art style, and get an AI-generated explanation. Please allow a moment for the analysis.",
    examples=[
        ["examples/starry_night.jpg"],
        ["examples/mona_lisa.jpg"],
        ["examples/the_persistence_of_memory.jpg"],
        ["examples/the_calling_of_saint_matthew.jpeg"],
        ["examples/pollock_number_1.jpeg"]
    ]
)

# Run the app
if __name__ == "__main__":
    demo.launch()
