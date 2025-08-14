# AI Art Advisor

An interactive web application that classifies the style of an artwork from an uploaded image and provides an AI-generated explanation about the style, its history, and key artists.

### Live Demo

You can try the AI Art Advisor live on Hugging Face Spaces:
[**https://huggingface.co/spaces/heejeyoo/ai-art-classifier**](https://huggingface.co/spaces/heejeyoo/ai-art-classifier)

## Project Overview

This project combines deep learning for feature extraction with classical machine learning for classification to identify the art style of a painting. It leverages a pre-trained `EfficientNet` model to extract powerful visual features from artworks, which are then fed into a Support Vector Machine (SVM) classifier. The final application is deployed as a user-friendly Gradio web app, which uses the Gemini API to generate rich, contextual explanations for the predicted art style.

## Features

* **Art Style Classification:** Predicts one of 15 major art styles from an uploaded image.
* **Feature Extraction:** Uses `EfficientNetB7` pre-trained on ImageNet for high-quality feature extraction.
* **Optimized Classifier:** Employs a Support Vector Machine (SVM) with hyperparameters tuned via `GridSearchCV` for best performance.
* **AI-Generated Explanations:** Leverages the Gemini API to provide engaging, museum-guide-style narratives about the predicted art style.
* **Interactive Web UI:** Built with Gradio for an easy-to-use, drag-and-drop interface.

## Tech Stack

* **Backend & ML:** Python, TensorFlow/Keras, Scikit-learn, Pandas
* **Feature Extractor:** `EfficientNetB7`
* **Classifier:** Support Vector Machine (SVC)
* **Deployment:** Gradio, Hugging Face Spaces
* **Generative AI:** Google Gemini API
* **Data Source:** [WikiArt Dataset on Kaggle](https://www.kaggle.com/datasets/antoinegruson/wikiart-art-pieces-classifiaction)

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://huggingface.co/spaces/heejeyoo/ai-art-classifier
    cd ai-art-classifier
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your Gemini API Key:**
    The application uses the Gemini API for text generation. You need to set your API key as an environment variable.
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

4.  **Download the necessary files:**
    Ensure the following files (generated from the Jupyter Notebook) are in the root directory:
    * `best_model.pkl`
    * `scaler.pkl`
    * `label_encoder.pkl`

5.  **Run the Gradio app:**
    ```bash
    python app.py
    ```
    The application will be available at a local URL (e.g., `http://127.0.0.1:7860`).

## Usage

1.  Navigate to the deployed Hugging Face Space or your local URL.
2.  Drag and drop an image of a painting into the input box, or click to upload.
3.  Allow a moment for the model to process the image and generate the explanation.
4.  The predicted art style will appear in the "Predicted Style" label.
5.  An AI-generated summary about the style will be displayed in the "About the Style" section.

## Project Structure

```
.
├── app.py                  # The main Gradio deployment script
├── AI_Art_Advisor.ipynb    # Jupyter Notebook for data processing, training, and evaluation
├── best_model.pkl          # Saved trained SVM model
├── scaler.pkl              # Saved StandardScaler object
├── label_encoder.pkl       # Saved LabelEncoder object
├── examples/               # Folder with example images for the Gradio interface
└── requirements.txt        # Python dependencies
```

## Acknowledgments

This project uses the **WikiArt dataset**, which contains over 80,000 high-quality images of paintings. The dataset is available on [Kaggle](https://www.kaggle.com/datasets/antoinegruson/wikiart-art-pieces-classifiaction).
