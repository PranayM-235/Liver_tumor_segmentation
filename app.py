import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model import UNet
from predict import predict

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Liver Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = UNet().to(device)
    model.load_state_dict(
        torch.load("UNet_final_model.pth", map_location=device)
    )
    model.eval()
    return model

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 About")

st.sidebar.info("""
This application uses a U-Net deep learning model
to detect possible liver tumor regions from CT scan images.
""")

st.sidebar.success("Model Loaded Successfully ✅")

# ---------------- HEADER ----------------
st.title("🧠 Liver Tumor Segmentation System")

st.markdown("""
Upload a CT scan image to detect possible liver tumor regions using AI.
""")

st.divider()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload CT Scan Image",
    type=["png", "jpg", "jpeg"]
)

# ---------------- PREDICTION ----------------
if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded CT Scan", width=300)

    st.write("")

    if st.button("🔍 Detect Tumor"):

        with st.spinner("Analyzing CT Scan..."):

            original, prob, pred = predict(image, model, device)

        # ---------------- OVERLAY ----------------
        overlay = original.copy()

        overlay = np.stack([overlay]*3, axis=-1)

        overlay[pred == 1] = [255, 0, 0]

        # ---------------- CONFIDENCE ----------------
        confidence = float(prob.max())

        # ---------------- LAYOUT ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼 Original CT Scan")
            st.image(original, clamp=True)

        with col2:
            st.subheader("🔥 Tumor Highlight")
            st.image(overlay, clamp=True)

        st.divider()

        # ---------------- METRICS ----------------
        metric1, metric2 = st.columns(2)

        tumor_pixels = int((prob > 0.2).sum())

        with metric1:
            st.metric("Tumor Pixels", tumor_pixels)

        with metric2:
            st.metric("Max Confidence", f"{confidence:.2f}")

        st.divider()

        # ---------------- STATUS ----------------
        if confidence < 0.3:
            st.success("✅ No Tumor Detected")

        elif confidence < 0.6:
            st.warning("⚠️ Suspicious region detected")

        else:
            st.error("🚨 High probability tumor detected. Consult a doctor.")

        # ---------------- OPTIONAL PROBABILITY MAP ----------------
        with st.expander("View Probability Heatmap"):

            fig, ax = plt.subplots()

            ax.imshow(prob, cmap="hot")
            ax.set_title("Tumor Probability Heatmap")
            ax.axis("off")

            st.pyplot(fig)