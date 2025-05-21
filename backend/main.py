from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Directories
UPLOAD_FOLDER = "uploads"
HEATMAP_FOLDER = "Heatmap"
MODEL_PATH = "model/cashew_classifier_model.h5"

# Ensure folders exist
for folder in [UPLOAD_FOLDER, HEATMAP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["HEATMAP_FOLDER"] = HEATMAP_FOLDER

# Load trained Keras model
model = tf.keras.models.load_model(MODEL_PATH)
class_labels = ["Discolored", "Jumbo", "Regular", "Special"]

def preprocess_image(image_path):
    """Preprocess image to match model input shape."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def generate_heatmap(image_path):
    """Generate and save a heatmap of the image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(5, 5))
    sns.heatmap(img, cmap="jet", xticklabels=False, yticklabels=False)

    # Create heatmap filename
    heatmap_filename = "heatmap_" + os.path.basename(image_path)
    heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)

    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()

    return heatmap_path

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Preprocess image
    processed_image = preprocess_image(file_path)

    # Predict using the model
    predictions = model.predict(processed_image)[0]
    predicted_class = class_labels[np.argmax(predictions)]
    confidence_scores = {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}

    # Generate heatmap
    heatmap_path = generate_heatmap(file_path)
    heatmap_url = f"http://127.0.0.1:5000/heatmap/{os.path.basename(heatmap_path)}"

    return jsonify({
        "predicted_class": predicted_class,
        "confidence_scores": confidence_scores,
        "image_url": f"http://127.0.0.1:5000/uploads/{filename}",
        "heatmap_url": heatmap_url
    })

# Route to serve uploaded images
@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Route to serve heatmap images
@app.route("/heatmap/<filename>")
def get_heatmap(filename):
    return send_from_directory(app.config["HEATMAP_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
