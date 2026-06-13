import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ✅ Model path (.h5 format)
model_path = "dr_model.h5"

# ✅ Download model if not exists
if not os.path.exists(model_path):
    st.info("📥 Downloading model... please wait")
    url = "https://drive.google.com/uc?id=1XijtAwHnkfpgsTA28afVUij8e-YaRNIL"
    try:
        gdown.download(url, model_path, quiet=False)
    except Exception as e:
        st.error(f"❌ Failed to download model: {str(e)}")
        st.stop()

# ✅ Load model (no compatibility issues with h5)
try:
    model = tf.keras.models.load_model(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

# ✅ UI Setup
st.title("Diabetic Retinopathy Detection")
st.markdown("Upload an eye fundus image to detect diabetic retinopathy severity level")

classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = image.resize((224, 224))
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Get prediction
        with st.spinner("Analyzing image..."):
            prediction = model.predict(img, verbose=0)
        
        result_idx = np.argmax(prediction)
        result_class = classes[result_idx]
        confidence = np.max(prediction) * 100

        # Display results
        st.success("✅ Analysis Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Class", result_class)
        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")
        
        # Display severity-based messaging
        st.subheader("Diagnosis:")
        if result_idx == 0:
            st.success("✅ **No Diabetic Retinopathy** - No signs of diabetic retinopathy detected.")
        elif result_idx == 1:
            st.warning("⚠️ **Mild Diabetic Retinopathy** - Early signs detected. Regular monitoring recommended.")
        elif result_idx == 2:
            st.warning("⚠️ **Moderate Diabetic Retinopathy** - Moderate signs detected. Consult an ophthalmologist.")
        elif result_idx == 3:
            st.error("🔴 **Severe Diabetic Retinopathy** - Severe signs detected. Immediate medical attention required.")
        else:
            st.error("🔴 **Proliferative Diabetic Retinopathy** - Advanced condition detected. Urgent medical intervention needed.")
        
        # Display all prediction probabilities
        st.subheader("All Predictions:")
        for i, class_name in enumerate(classes):
            st.write(f"**{class_name}**: {prediction[0][i]*100:.2f}%")
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
