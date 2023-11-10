# app.py
import streamlit as st
from PIL import Image
from ultralytics import YOLO  
import numpy as np

model = YOLO("best.pt")

st.title("YOLOv8 Race Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)



    if st.button("Predict"):
        results = model(image)
        probs = results[0].probs.data.tolist()
        names = results[0].names
        prediction = names[np.argmax(probs)]
        st.success(f"The Prediction is {prediction}")
