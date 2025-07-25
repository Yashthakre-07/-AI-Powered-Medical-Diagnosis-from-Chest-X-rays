# ==============================================================================
# Flask Backend for AI-Powered Pneumonia Detection
#
# This script creates a web server that:
# 1. Loads the pre-trained Keras model.
# 2. Provides a main page to upload an X-ray image.
# 3. Creates a `/predict` API endpoint to receive the image,
#    preprocess it, and return a JSON response with the diagnosis.
#
# To Run:
# 1. Make sure you have Flask, TensorFlow, and Pillow installed:
#    pip install Flask tensorflow Pillow numpy
# 2. Place this file in your project folder.
# 3. Place your `pneumonia_mobilenetv2_model.h5` file in the same folder.
# 4. Create a 'templates' folder and put `index.html` inside it.
# 5. Run the script from your terminal: `python app.py`
# 6. Open your browser and go to http://127.0.0.1:5000
# ==============================================================================

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Model and Image Configuration ---
MODEL_PATH = 'pneumonia_mobilenetv2_model.h5'
IMG_WIDTH, IMG_HEIGHT = 160, 160
CLASS_LABELS = ['Normal', 'Pneumonia']

# Load the trained model
# We use a try-except block to handle potential errors during model loading.
try:
    print("Loading the model... Please wait.")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # If the model can't be loaded, we can't proceed.
    # In a real application, you might have a fallback or exit gracefully.
    model = None

def preprocess_image(img_file):
    """
    Loads an image file from the request and preprocesses it for the model.
    """
    try:
        # Load the image from the file stream
        img = Image.open(io.BytesIO(img_file.read()))

        # Convert grayscale images to RGB, as the model expects 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize the image to the target dimensions
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))

        # Convert the image to a NumPy array
        img_array = image.img_to_array(img)

        # Expand dimensions to create a batch of 1
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image using the same function as during training
        preprocessed_img = preprocess_input(img_array)

        return preprocessed_img
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle the image upload and return the prediction.
    """
    if model is None:
        return jsonify({'error': 'Model is not loaded, cannot make a prediction.'}), 500

    # Check if a file was posted
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            # Preprocess the image
            processed_image = preprocess_image(file)
            if processed_image is None:
                return jsonify({'error': 'Could not process the image.'}), 400

            # Make a prediction
            prediction = model.predict(processed_image)
            
            # Get the confidence score (probability)
            score = float(prediction[0][0])
            
            # Determine the class based on the score
            if score > 0.5:
                predicted_class = CLASS_LABELS[1] # Pneumonia
                confidence = score * 100
            else:
                predicted_class = CLASS_LABELS[0] # Normal
                confidence = (1 - score) * 100

            # Return the result as JSON
            return jsonify({
                'prediction': predicted_class,
                'confidence': f"{confidence:.2f}"
            })

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': 'An error occurred during prediction.'}), 500

    return jsonify({'error': 'An unknown error occurred.'}), 500

# --- Main execution ---
if __name__ == '__main__':
    # The app runs on http://127.0.0.1:5000 by default
    app.run(debug=True)
