import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
    # Convert to HSV and extract color features
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    color_features = [
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ]
    
    # Convert to grayscale and extract texture features using GLCM
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    texture_features = [
        graycoprops(glcm, prop).flatten()[0] for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    ]
    
    # Convert to binary image for shape analysis
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract shape features if contour exists
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, width, height = cv2.boundingRect(cnt)
        aspect_ratio = width / height
        convex_hull = cv2.convexHull(cnt)
        convex_hull_area = cv2.contourArea(convex_hull)
        convexity = area / convex_hull_area if convex_hull_area > 0 else 0
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    else:
        area, perimeter, aspect_ratio, convexity, circularity, width, height = 0, 0, 0, 0, 0, 0, 0

    shape_features = [area, perimeter, aspect_ratio, convexity, circularity, width, height]
    
    return color_features + texture_features + shape_features

# Process the dataset
accuracy_per_directory = {}
for split in ['train', 'validation']:
    split_dir = os.path.join(data_dir, split)
    for label_idx, label in enumerate(classes):
        class_dir = os.path.join(split_dir, label)
        class_features = []
        class_labels = []
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                image = cv2.imread(file_path)
                image = cv2.resize(image, (224, 224))
                
                extracted_features = extract_features(image)
                class_features.append(extracted_features)
                class_labels.append(label_idx)
                
                features.append(extracted_features)
                labels.append(label_idx)
        
        if class_features:
            class_features = np.array(class_features)
            class_labels = np.array(class_labels)
            X_train, X_test, y_train, y_test = train_test_split(class_features, class_labels, test_size=0.2, random_state=42)
            class_model = RandomForestClassifier(n_estimators=100, random_state=42)
            class_model.fit(X_train, y_train)
            class_accuracy = accuracy_score(y_test, class_model.predict(X_test))
            accuracy_per_directory[label] = class_accuracy

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
train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=classes))
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

