import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import random

# Page config
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# Title
st.title("🧠 Brain Tumor Detection")
st.markdown("This is a simulated version of a brain tumor detection system. Upload an MRI scan to see the result.")

# Image uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Resize and normalize
    img = ImageOps.fit(image, (224, 224), method=Image.Resampling.LANCZOS)
    img_array = np.asarray(img) / 255.0

    # Predict button
    if st.button("Simulate Prediction"):
        confidence = random.uniform(0.4, 0.99)  # Simulated confidence
        tumor_detected = confidence > 0.5

        if tumor_detected:
            st.error(f"🧠 Tumor Detected with {confidence*100:.2f}% confidence")
        else:
            st.success(f"✅ No Tumor Detected with {(1 - confidence)*100:.2f}% confidence")

        st.markdown("---")
        st.markdown("**Note**: This is a simulation. Replace logic with real model inference for production use.")

        st.markdown("---")
        st.markdown("Developed by ShravaniJ~.")
