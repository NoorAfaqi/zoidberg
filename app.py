import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import json

from preprocessing import val_test_transforms as transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Model paths
MODEL_PATHS = {
    "ResNet50": "resnet50_pneumonia.pth",
    "EfficientNet-B4": "EfficientNet-B4_pneumonia.pth",
    "ConvNeXt": "ConvNeXt_pneumonia.pth"
}

CLASS_NAMES_PATH = "classes.json"


# ------------------ LOAD CLASSES ------------------
@st.cache_resource
def load_classes():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

class_names = load_classes()


# ------------------ MODEL FACTORY ------------------
def build_model(model_name, pretrained=False):
    model_name = model_name.lower()
    num_classes = 3
    if model_name == "resnet50":
        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained else None
        )
        model = models.resnet50(
            weights=weights
        )
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == "efficientnet":
        weights = (
            models.EfficientNet_B4_Weights.DEFAULT
            if pretrained else None
        )
        model = models.efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    elif model_name == "convnext":
        weights = (
            models.ConvNeXt_Base_Weights.DEFAULT
            if pretrained else None
        )
        model = models.convnext_base(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model(model_name):
    model = build_model(model_name)

    model.load_state_dict(
        torch.load(MODEL_PATHS[model_name], map_location=DEVICE)
    )

    model = model.to(DEVICE)
    model.eval()
    return model


# ------------------ UI ------------------
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.title("🧠 Pneumonia Detection")
st.write("Upload a chest X-ray image and select a model to predict.")

# ✅ Model selector
selected_model_name = st.selectbox(
    "Choose Model",
    list(MODEL_PATHS.keys())
)

model = load_model(selected_model_name)

st.write(f"Using model: **{selected_model_name}**")


# ------------------ IMAGE INPUT ------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="content")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)

        confidence, pred_class = torch.max(probs, 1)

    predicted_label = class_names[pred_class.item()]

    st.success(f"Prediction: {predicted_label}")
    st.progress(float(confidence))

    # 🔥 Top predictions
    st.subheader("Top Predictions")
    top_k = min(3, len(class_names))
    top_prob, top_idx = torch.topk(probs, top_k)

    for i in range(top_k):
        label = class_names[top_idx[0][i].item()]
        prob = top_prob[0][i].item() * 100
        st.write(f"{label}: {prob:.2f}%")

st.markdown("---")
st.caption("Powered by PyTorch + Streamlit")