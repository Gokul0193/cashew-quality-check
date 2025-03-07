import cv2
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import graycomatrix, graycoprops

# Define paths
model_filename = "model/prediction_train_model2.joblib"  # Ensure the model is inside the "model/" directory
uploads_dir = "uploads"  # Directory where uploaded images are stored
classes = ["Discolored", "Jumbo", "Regular", "Special"]  # Class names

# Helper function to extract features
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

# Load the trained model
model = joblib.load(model_filename)

# Function to generate probability graph
def generate_probability_graph(probabilities):
    plt.figure(figsize=(6, 4))
    plt.bar(classes, probabilities, color=['red', 'green', 'blue', 'purple'])
    plt.xlabel("Cashew Quality Classes")
    plt.ylabel("Prediction Probability")
    plt.title("Prediction Probability Distribution")

    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    graph_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()
    return graph_base64

# Function to predict the class of an image
def predict_image(image_path):
    if not os.path.exists(image_path):
        return {"error": "File not found"}

    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Image could not be loaded"}

    image_resized = cv2.resize(image, (224, 224))  # Resize to standard dimensions

    # Extract features
    features = extract_features(image_resized)
    print(f"Extracted {len(features)} features.")  # Debugging print

    # Check feature shape
    expected_features = model.n_features_in_
    if len(features) != expected_features:
        return {"error": f"Feature length mismatch. Expected {expected_features}, got {len(features)}"}

    # Make prediction
    prediction = model.predict([features])[0]
    predicted_class = classes[prediction]

    # Get prediction probabilities
    probabilities = model.predict_proba([features])[0]
    probabilities_dict = dict(zip(classes, probabilities))

    # Generate probability graph
    graph_base64 = generate_probability_graph(probabilities)

    return {
        "image": os.path.basename(image_path),
        "predicted_class": predicted_class,
        "class_probabilities": probabilities_dict,
        "graph": graph_base64
    }
