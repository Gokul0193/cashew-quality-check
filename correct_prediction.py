import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import graycomatrix, graycoprops
import os
import joblib
import matplotlib.pyplot as plt

# Define paths for dataset
model_filename = 'backend/model/new_prediction_train_model2.joblib'  # Corrected model path
prediction_dir = 'prediction'  # Directory containing images to predict
classes = ["Discolored", "Jumbo", "Regular", "Special"]  # Class names

# Check if the model file exists
if not os.path.exists(model_filename):
    print(f"Error: Model file '{model_filename}' not found! Make sure it exists and is in the correct directory.")
    exit()

# Load the trained model
model = joblib.load(model_filename)

# Helper function to extract features
def extract_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    # Color features
    color_features = [
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ]
    
    # Texture features using GLCM
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    texture_features = [
        graycoprops(glcm, prop).flatten()[0] for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    ]
    
    return color_features + texture_features

# Function to make predictions on images in the given directory
def predict_images_in_directory(prediction_dir):
    if not os.path.exists(prediction_dir):
        print(f"Error: Prediction directory '{prediction_dir}' does not exist.")
        return
    
    for file_name in os.listdir(prediction_dir):
        file_path = os.path.join(prediction_dir, file_name)
        
        if file_name.lower().endswith(('png', 'jpg', 'jpeg')):  # Process image files only
            # Load and display image
            image = cv2.imread(file_path)
            image_resized = cv2.resize(image, (224, 224))  # Resize to standard dimensions
            
            plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))  # Show image using Matplotlib
            plt.axis('off')
            plt.show()
            
            # Extract features from the image
            features = extract_features(image_resized)
            
            # Make prediction
            prediction = model.predict([features])  # Model expects a 2D array
            predicted_class = classes[prediction[0]]
            
            # Get prediction probabilities
            probabilities = model.predict_proba([features])[0]  # Probabilities for each class
            
            # Plot bar graph for probabilities
            plt.bar(classes, probabilities, color='blue')
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title(f'Prediction for {file_name}: {predicted_class}')
            plt.ylim([0, 1])
            plt.show()
            
            # Print prediction result
            print(f"Image: {file_name}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Class Probabilities: {dict(zip(classes, probabilities))}")
            print("-" * 50)

# Run the prediction on all images in the 'prediction' directory
predict_images_in_directory(prediction_dir)
