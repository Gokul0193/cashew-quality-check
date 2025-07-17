from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, numpy as np, cv2, matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
HEATMAP_FOLDER = "Heatmap"
MODEL_PATH = "model/cashew_classifier_model.h5"

for folder in [UPLOAD_FOLDER, HEATMAP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["HEATMAP_FOLDER"] = HEATMAP_FOLDER

model = tf.keras.models.load_model(MODEL_PATH)
class_labels = ["Discolored", "Jumbo", "Regular", "Special"]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def generate_heatmap(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(5, 5))
    sns.heatmap(img, cmap="jet", xticklabels=False, yticklabels=False)
    heatmap_filename = "heatmap_" + os.path.basename(image_path)
    heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()
    return heatmap_filename

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

    processed_image = preprocess_image(file_path)
    predictions = model.predict(processed_image)[0]
    predicted_class = class_labels[np.argmax(predictions)]
    confidence_scores = {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}

    heatmap_filename = generate_heatmap(file_path)

    return jsonify({
        "predicted_class": predicted_class,
        "confidence_scores": confidence_scores,
        "image_url": f"/uploads/{filename}",
        "heatmap_url": f"/heatmap/{heatmap_filename}"
    })

@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/heatmap/<filename>")
def get_heatmap(filename):
    return send_from_directory(app.config["HEATMAP_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
