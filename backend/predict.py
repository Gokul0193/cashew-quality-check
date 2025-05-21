import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
from flask import request, jsonify

UPLOAD_FOLDER = "uploads"
HEATMAP_FOLDER = "Heatmap"

# Ensure Heatmap folder exists
if not os.path.exists(HEATMAP_FOLDER):
    os.makedirs(HEATMAP_FOLDER)

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

def predict():
    # File handling
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Generate heatmap
    heatmap_path = generate_heatmap(file_path)
    heatmap_url = f"http://127.0.0.1:5000/heatmap/{os.path.basename(heatmap_path)}"

    return jsonify({
        "image_url": f"http://127.0.0.1:5000/uploads/{filename}",
        "heatmap_url": heatmap_url
    })
