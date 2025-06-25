import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load trained model
model = load_model("mnist_cnn_model.h5")

st.title("MNIST Digit Classifier")
st.write("Upload a 28x28 grayscale image of a handwritten digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)  # Invert for white digits on black background
    image = image.resize((28, 28))
    st.image(image, caption='Uploaded Image', use_column_width=False)

    # Preprocess image
    img = np.array(image).reshape(1, 28, 28, 1) / 255.0

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    st.write(f"### Predicted Digit: {predicted_class}")
