import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained CNN model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.title("🧠 MNIST Digit Classifier")
st.write("Upload a 28x28 grayscale PNG image of a handwritten digit.")

uploaded_file = st.file_uploader("Choose PNG image", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 like MNIST
    img_array = np.array(image).reshape(1, 28, 28) / 255.0

    st.image(image, caption="Uploaded Image", use_column_width=False)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.success(f"✅ Predicted Digit: {predicted_class}")
