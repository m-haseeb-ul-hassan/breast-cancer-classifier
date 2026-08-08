# streamlit_app.py
# this replaces the gradio app.py -- same model, same predict logic, different UI framework
# deploy this on Streamlit Community Cloud (share.streamlit.io), completely free, unaffected
# by Hugging Face's recent Gradio/Docker paywall change

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
CROP_W, CROP_H = 560, 368  # must match the moderate-crop setup used in training
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


# cache the model so it only loads once, not on every interaction (streamlit reruns the
# whole script top-to-bottom on every button click, this decorator prevents reloading each time)
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 2)
    )
    model.load_state_dict(torch.load("finetuned_model_cropped.pt", map_location=device))
    model = model.to(device)
    model.eval()

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    return model, cam


def prep_image(pil_image):
    img = np.array(pil_image.convert("RGB"))
    h, w, _ = img.shape

    if h < CROP_H or w < CROP_W:
        scale = max(CROP_H / h, CROP_W / w) + 0.01
        img = cv2.resize(img, (int(w * scale) + 1, int(h * scale) + 1))
        h, w, _ = img.shape

    top = (h - CROP_H) // 2
    left = (w - CROP_W) // 2
    img = img[top:top + CROP_H, left:left + CROP_W]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_float = img.astype(np.float32) / 255.0
    normalized = (img_float - MEAN) / STD
    tensor = torch.tensor(normalized.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    return img_float, tensor


def predict(model, cam, pil_image, threshold):
    img_float, tensor = prep_image(pil_image)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)
        prob_malignant = torch.softmax(output, dim=1)[0, 1].item()

    pred_class = 1 if prob_malignant >= threshold else 0
    label = "Malignant" if pred_class == 1 else "Benign"

    targets = [ClassifierOutputTarget(1)]  # heatmap always explains "what looks malignant"
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    overlay = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    return label, prob_malignant, (img_float * 255).astype(np.uint8), overlay


# ---------------- UI ----------------
st.set_page_config(page_title="Breast Cancer Histopathology Classifier", page_icon="🔬", layout="wide")

st.title("🔬 Breast Cancer Histopathology Classifier")
st.write("Upload a breast tissue histopathology image. The model predicts Benign vs Malignant and shows a Grad-CAM heatmap of what it focused on.")

st.warning(
    "**This is a student research/portfolio project, not a medical device.** "
    "It is not validated for clinical use and should never be used to make real diagnostic decisions. "
    "Trained on the public BreaKHis histopathology dataset for educational purposes only."
)

model, cam = load_model()

col_input, col_output = st.columns([1, 2])

with col_input:
    uploaded_file = st.file_uploader("Upload histopathology image", type=["png", "jpg", "jpeg"])
    threshold = st.slider(
        "Malignant decision threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05,
        help="Lower = catches more cancer cases but more false alarms. 0.35 is the clinically-motivated default from our evaluation."
    )
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

with col_output:
    if uploaded_file is not None and analyze_btn:
        pil_image = Image.open(uploaded_file)
        label, prob, cropped_img, heatmap = predict(model, cam, pil_image, threshold)

        confidence = prob if label == "Malignant" else (1 - prob)
        if label == "Malignant":
            st.error(f"**Prediction: {label}**  (P(malignant) = {prob:.1%}, threshold = {threshold:.2f})")
        else:
            st.success(f"**Prediction: {label}**  (P(malignant) = {prob:.1%}, threshold = {threshold:.2f})")

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(cropped_img, caption="Input (cropped/resized)", use_container_width=True)
        with img_col2:
            st.image(heatmap, caption="Grad-CAM heatmap", use_container_width=True)
    elif uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image (click Analyze)", use_container_width=True)
    else:
        st.info("Upload an image and click Analyze to see results.")

st.markdown("---")
st.caption("Model: EfficientNetB0 (transfer learning, fine-tuned last block) | Dataset: BreaKHis | Explainability: Grad-CAM")
