from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid thread issues
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from skimage.feature import graycomatrix, graycoprops
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Ensure the uploads directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = "model/new_prediction_train_model.joblib"
model = joblib.load(MODEL_PATH)

# Class labels
classes = ["Discolored", "Jumbo", "Regular", "Special"]

# Function to extract features from an image
def extract_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    color_features = [
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ]
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    texture_features = [
        graycoprops(glcm, prop).flatten()[0] for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    ]
    
    return color_features + texture_features

# Function to generate probability graph
def generate_probability_graph(probabilities):
    plt.figure(figsize=(6, 4))
    plt.bar(classes, probabilities, color=['red', 'green', 'blue', 'purple'])
    plt.xlabel("Cashew Quality Classes")
    plt.ylabel("Prediction Probability")
    plt.title("Prediction Probability Distribution")
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    graph_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()
    return graph_base64

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Cashew Quality Prediction API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read and preprocess the image
    image = cv2.imread(file_path)
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    image_resized = cv2.resize(image, (224, 224))  # Resize to standard dimensions

    # Extract features
    features = extract_features(image_resized)

    # Make prediction
    prediction = model.predict([features])[0]
    predicted_class = classes[prediction]

    # Get prediction probabilities
    probabilities = model.predict_proba([features])[0]
    probabilities_dict = dict(zip(classes, probabilities))

    # Generate probability graph
    graph_base64 = generate_probability_graph(probabilities)

    return jsonify({
        "image": file.filename,
        "predicted_class": predicted_class,
        "class_probabilities": probabilities_dict,
        "graph": graph_base64
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
