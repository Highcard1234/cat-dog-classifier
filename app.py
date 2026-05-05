import os
import gdown
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

@st.cache_resource
def load_model():
    if not os.path.exists("model.h5"):
        url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
        gdown.download(url, "model.h5", quiet=False)
    return tf.keras.models.load_model("model.h5")
model = load_model()

st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("Prediction: DOG 🐶")
    else:
        st.success("Prediction: CAT 🐱")