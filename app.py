import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('brain_tumor_model.h5')
    return model

model = load_model()

# Page settings
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# Title
st.title("🧠 Brain Tumor Detection")
st.markdown("Upload an MRI scan image and the model will predict whether a brain tumor is present or not.")

# Image uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess image
    img = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)  # Resize to model input
    img_array = np.asarray(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict button
    if st.button("Predict"):
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        if confidence > 0.5:
            st.error(f"🧠 Tumor Detected with {confidence*100:.2f}% confidence")
        else:
            st.success(f"✅ No Tumor Detected with {(1 - confidence)*100:.2f}% confidence")

        st.markdown("---")
        st.markdown("**Note**: This tool is for educational purposes and not for clinical diagnosis.")
