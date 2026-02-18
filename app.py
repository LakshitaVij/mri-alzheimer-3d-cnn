import streamlit as st
import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from model import Simple3DCNN
import tempfile
import os

# ---- Load trained model once ----
@st.cache_resource #“Load this once and reuse it.”
def load_model():
    model = Simple3DCNN()
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("🧠 3D MRI Alzheimer Prediction Demo")

uploaded_file = st.file_uploader(
    "Upload MRI (.mgz, .nii, .nii.gz)", 
    type=["mgz", "nii", "gz"]
)

if uploaded_file is not None:

    # ---- Preserve original file extension ----
    uploaded_file.seek(0)
    filename = uploaded_file.name
    
    # Correct handling for .nii.gz
    if filename.endswith(".nii.gz"):
        suffix = ".nii.gz"
    else:
        suffix = os.path.splitext(filename)[1]

    # ---- Write temp file with correct suffix ----
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ---- Load MRI correctly ----
    image = nib.load(tmp_path)

    # Optional: standardize orientation 
    image = nib.as_closest_canonical(image)

    volume = image.get_fdata()

    # ---- Normalize safely ----
    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)

    # ---- Convert to tensor ----
    volume = torch.tensor(volume, dtype=torch.float32)

    # Shape should be (D, H, W)
    volume = volume.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

    # ---- Resize to (128,128,128) ----
    volume = F.interpolate(
        volume,
        size=(128, 128, 128),
        mode="trilinear",
        align_corners=False
    )

    # ---- Inference ----
    with torch.no_grad():
        output = model(volume)
        prob = torch.sigmoid(output).item()

    # ---- Display result ----
    st.subheader("Prediction Result")
    st.write(f"Probability of Alzheimer: {prob:.4f}")

    if prob > 0.5:
        st.error("Predicted: Alzheimer")
    else:
        st.success("Predicted: Healthy")

    # ---- Cleanup temp file ----
    os.remove(tmp_path)