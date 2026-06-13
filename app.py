import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ✅ ADD THIS PART HERE (top of file or before model load)
model_path = "dr_model.keras"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1oKj-DjI3Y2CdvIeAC3YgpXMYZLhI9jn0"
    gdown.download(url, model_path, quiet=False, fuzzy=True)

# ✅ THEN load model
model = tf.keras.models.load_model(model_path)

# --- UI PART ---
st.title("Diabetic Retinopathy Detection")

classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    result = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.write("Prediction:", result)
    st.write(f"Confidence: {confidence:.2f}%")
