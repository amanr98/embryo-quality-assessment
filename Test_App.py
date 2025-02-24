# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:02:39 2025

@author: amanr
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

# Class Labels Mapping
class_labels = {
    0: '8 Cell A', 1: '8 Cell B', 2: '8 Cell C',
    3: 'Blastocyst A', 4: 'Blastocyst B', 5: 'Blastocyst C',
    6: 'Morula A', 7: 'Morula B', 8: 'Morula C'
}

# Load Trained Model
MODEL_PATH = r"E:\360_project\Models\final_xception_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

st.title("Embryo Image Classifier")

st.markdown("""
    <style>
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #ff5733;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stError {
        background-color: #FFC0CB;
        color: #FF0000;
        font-weight: bold;
    }
    .stPrediction {
        background-color: #E0FFFF;
        color: #008080;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    .stConfidence {
        color: #32CD32;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    .fileUploader {
        background-color: #87CEFA;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "webp"])

def is_black_and_white(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(img, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    return np.count_nonzero(difference) == 0

def predict_single_image(img):
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return class_labels[predicted_class], confidence

if uploaded_file is not None:
    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True, output_format="JPEG")

    if is_black_and_white(img):
        pil_img = Image.open(uploaded_file).resize((224, 224))
        label, conf = predict_single_image(pil_img)
        with col2:
            st.markdown(f"<div class='stPrediction'>**Predicted Label:** {label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stConfidence'>**Confidence:** {conf * 100:.2f}%</div>", unsafe_allow_html=True)
    else:
        st.error("This is not a embryo image.")
