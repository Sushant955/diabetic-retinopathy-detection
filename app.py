import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import warnings

warnings.filterwarnings('ignore')

# ✅ Model path
model_path = "dr_model.keras"

# ✅ Download model if not exists
if not os.path.exists(model_path):
    st.info("📥 Downloading model... this may take a minute")
    url = "https://drive.google.com/uc?id=1oKj-DjI3Y2CdvIeAC3YgpXMYZLhI9jn0"
    try:
        gdown.download(url, model_path, quiet=False)
    except Exception as e:
        st.error(f"❌ Failed to download model: {str(e)}")
        st.stop()

# ✅ Verify file exists and has content
if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
    st.error("❌ Model file is missing or corrupted. Please try again or check the download link.")
    st.stop()

# ✅ Load model with improved error handling
def load_model_with_fallback():
    """
    Try multiple approaches to load the model with version compatibility
    """
    try:
        # First attempt: Direct load with compile=False to skip potential issues
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e1:
        st.warning(f"⚠️ Standard load failed: {str(e1)[:100]}... Trying alternative method...")
        try:
            # Second attempt: Load with custom objects
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=None,
                compile=False
            )
            return model
        except Exception as e2:
            st.warning(f"⚠️ Custom load failed. Attempting h5 conversion...")
            try:
                # Third attempt: Try loading as h5 if file is actually h5
                model = tf.keras.models.load_model(model_path.replace('.keras', '.h5'), compile=False)
                return model
            except Exception as e3:
                raise Exception(f"All model loading attempts failed:\n1. {str(e1)}\n2. {str(e2)}\n3. {str(e3)}")

try:
    model = load_model_with_fallback()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.info("""
    **💡 Solutions:**
    1. Update TensorFlow: `pip install --upgrade tensorflow`
    2. Update to TensorFlow 2.13+: `pip install tensorflow>=2.13`
    3. Ensure model file is not corrupted
    4. Check that the download link is still valid
    """)
    st.stop()

# --- UI PART ---
st.title("Diabetic Retinopathy Detection")
st.markdown("Upload an eye fundus image to detect diabetic retinopathy severity level")

classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = image.resize((224, 224))
        
        # Convert to RGB if necessary (handles grayscale images)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        with st.spinner("Analyzing image..."):
            prediction = model.predict(img, verbose=0)
        
        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        st.success("✅ Analysis Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Class", result)
        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")
        
        st.subheader("All Predictions:")
        for i, class_name in enumerate(classes):
            st.write(f"**{class_name}**: {prediction[0][i]*100:.2f}%")
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
