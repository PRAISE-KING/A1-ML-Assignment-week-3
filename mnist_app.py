import streamlit as st
from PIL import Image

st.title("🧠 MNIST Digit Classifier (Demo)")
st.write("Upload a digit image (PNG), and I'll pretend to predict it 😉")

uploaded_file = st.file_uploader("Choose a PNG image of a handwritten digit", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Digit", use_column_width=False)
    
    # Simulated prediction result
    st.success("✅ Predicted Digit: 5 (demo result)")
