import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ✅ Model path
model_path = "dr_model.keras"

# ✅ Download model if not exists (removed fuzzy=True to avoid errors)
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1oKj-DjI3Y2CdvIeAC3YgpXMYZLhI9jn0"
    gdown.download(url, model_path, quiet=False)

# ✅ Load model
model = tf.keras.models.load_model(model_path)

# ✅ UI Configuration
st.title("Diabetic Retinopathy Detection")
st.markdown("Upload an eye fundus image to detect diabetic retinopathy severity level")

# Class labels
classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

# File uploader (added jpeg support)
uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    
    # Convert to RGB if necessary (handles grayscale images)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Display results
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Detected Class", classes[predicted_class])
    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    # Detailed results with severity indicators
    if predicted_class == 0:
        st.success("✅ No Diabetic Retinopathy Detected")
    elif predicted_class == 1:
        st.warning("⚠️ Mild Diabetic Retinopathy Detected")
    elif predicted_class == 2:
        st.warning("⚠️ Moderate Diabetic Retinopathy Detected")
    elif predicted_class == 3:
        st.error("❌ Severe Diabetic Retinopathy Detected")
    else:
        st.error("❌ Proliferative Diabetic Retinopathy Detected")
    
    # Display prediction probabilities
    st.subheader("All Predictions")
    for i, class_name in enumerate(classes):
        st.write(f"{class_name}: {prediction[0][i] * 100:.2f}%")
