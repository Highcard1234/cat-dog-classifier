import os
import gdown
import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

@st.cache_resource
def load_model():
    if not os.path.exists("model.pth"):
        url = "https://drive.google.com/uc?id=13EcgDb9tpTAVesfU3vofAYlST3Xuj3Gz"
        try:
            gdown.download(url, "model.pth", quiet=False)
        except:
            st.warning("Could not download model. Using pre-trained ResNet50 instead.")
            model = models.resnet50(pretrained=True)
            return model.eval()
    
    try:
        model = torch.load("model.pth", map_location=torch.device('cpu'))
        return model.eval()
    except:
        # Fallback to pre-trained model
        model = models.resnet50(pretrained=True)
        return model.eval()

model = load_model()
device = torch.device('cpu')

# Display student information
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Name", "Elefue Divine")
with col2:
    st.metric("Reg Number", "20231372805")
with col3:
    st.metric("Course Code", "CSC 309")
st.markdown("---")

st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)
        probabilities = torch.nn.functional.softmax(prediction, dim=1)
        confidence = probabilities.max().item()

    if probabilities[0][1] > 0.5:
        st.success(f"Prediction: DOG 🐶 (Confidence: {confidence:.2%})")
    else:
        st.success(f"Prediction: CAT 🐱 (Confidence: {confidence:.2%})")
