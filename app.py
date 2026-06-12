import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.title("Diabetic Retinopathy Detection")

# Download model from Google Drive
model_path = "dr_model.keras"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1oKj-DjI3Y2CdvIeAC3YgpXMYZLhI9jn0"
    gdown.download(url, model_path, quiet=False)

model = tf.keras.models.load_model(model_path)

classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    result = classes[np.argmax(pred)]
    conf = np.max(pred)*100

    st.write("Prediction:", result)
    st.write(f"Confidence: {conf:.2f}%")
