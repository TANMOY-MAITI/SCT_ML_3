import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


st.set_page_config(page_title="Dogs vs Cats Classifier", layout="centered")

st.title("🐶🐱 Dogs vs Cats Classifier")
st.markdown("Upload an image, and the model will predict whether it's a **dog** or a **cat**!")


@st.cache_resource
def load_selected_model():
    return load_model("dogs_vs_cats_model.keras")

model = load_selected_model()

st.markdown("### 📂 Drag & Drop or Browse to Upload Your Image")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing Image..."):
            prediction = model.predict(img_array)[0][0]
            confidence = round(prediction * 100, 2) if prediction >= 0.5 else round((1 - prediction) * 100, 2)
            label = "🐶 Dog" if prediction >= 0.5 else "🐱 Cat"

            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence:** {confidence}%")


