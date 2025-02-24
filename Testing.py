# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 18:22:57 2025

@author: amanr
"""

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Function to load and preprocess a single image
def load_and_preprocess_image(img_path, img_height, img_width):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to make predictions on a single image
def predict_image(model_path, img_path, img_height, img_width, class_names):
    model = tf.keras.models.load_model(model_path)
    img_array = load_and_preprocess_image(img_path, img_height, img_width)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    print(f"Predicted class: {class_names[predicted_class]} with confidence: {confidence * 100:.2f}%")

# Define image dimensions and model path
IMG_HEIGHT, IMG_WIDTH = 224, 224
model_path = r"E:\360_project\Models\final_xception_model.keras"

# Define the class names (you can get these from the training data)
class_names = ['Grade A', 'Grade_B', 'Grade_C']  # Replace with your actual class names

# Path to the test image
test_image_path = r"E:\360_project\Dataset\Test data\Test  (16).png"  # Replace with your actual test image path

# Predict the class of the test image
predict_image(model_path, test_image_path, IMG_HEIGHT, IMG_WIDTH, class_names)
