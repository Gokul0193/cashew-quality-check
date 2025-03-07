import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops
import os
import joblib

# Define paths for dataset
data_dir = "cashew_dataset"
classes = ["Discolored", "Jumbo", "Regular", "Special"]

# Initialize lists for features and labels
features = []
labels = []

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

# Process the dataset
for split in ['train', 'validation']:  # Loop through both 'train' and 'validation' directories
    split_dir = os.path.join(data_dir, split)
    for label_idx, label in enumerate(classes):
        class_dir = os.path.join(split_dir, label)
        print(f"Processing directory: {class_dir}")  # Debugging line to check the paths
        if not os.path.exists(class_dir):
            print(f"Directory does not exist: {class_dir}")
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if file_name.lower().endswith(('png', 'jpg', 'jpeg')):  # Process image files only
                # Load image
                image = cv2.imread(file_path)
                image = cv2.resize(image, (224, 224))  # Resize to standard dimensions

                # Extract features and append to lists
                features.append(extract_features(image))
                labels.append(label_idx)

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Save the trained model
model_filename = 'new_prediction_train_model.joblib'
joblib.dump(classifier, model_filename)
print(f"Model saved to {model_filename}")

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=classes))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to make prediction on new images
def predict_image(file_path):
    # Load image
    image = cv2.imread(file_path)
    image = cv2.resize(image, (224, 224))  # Resize to standard dimensions
    features = extract_features(image)

    # Load the trained model
    model = joblib.load(model_filename)

    # Predict using the model
    prediction = model.predict([features])  # The model expects a 2D array (1 sample, n features)
    print(f"Prediction: {classes[prediction[0]]}")

# Example usage: Make a prediction on a new image
new_image_path = 'path_to_your_image.jpg'  # Provide the path to the new image
predict_image(new_image_path)
