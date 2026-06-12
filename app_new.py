import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Diabetic Retinopathy Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dr_detection_model.keras")

model = load_model()

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
