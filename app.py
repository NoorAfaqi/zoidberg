import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import json

from preprocessing import val_test_transforms as transform

MODEL_PATH = "resnet50_pneumonia.pth"
CLASS_NAMES_PATH = "classes.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_classes():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

class_names = load_classes()

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.title("🧠 Pneumonia Detection (ResNet50)")
st.write("Upload a chest X-ray image and the model will predict if it's NORMAL or PNEUMONIA.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)

        confidence, pred_class = torch.max(probs, 1)

    predicted_label = class_names[pred_class.item()]

    st.success(f"Prediction: {predicted_label}")
    st.progress(float(confidence))

    # Top predictions
    st.subheader("Top Predictions")
    top_k = min(3, len(class_names))
    top_prob, top_idx = torch.topk(probs, top_k)

    for i in range(top_k):
        label = class_names[top_idx[0][i].item()]
        prob = top_prob[0][i].item() * 100
        st.write(f"{label}: {prob:.2f}%")

st.markdown("---")
st.caption("Powered by PyTorch + Streamlit")