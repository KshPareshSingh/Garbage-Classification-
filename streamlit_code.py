import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
from PIL import Image

EN_B2_model = tf.keras.models.load_model('Efficientnetv2b2.keras')
class_names = ['cardboard','glass','metal','paper','plastic','trash']

# Streamlit UI
st.title("Garbage Classifier")
st.write("Upload an image of garbage to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((160,160))  # Must match training input size
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = EN_B2_model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = prediction[0][predicted_class_index]

    # Show result
    st.write(f"### Predicted: **{predicted_class}**")
    st.write(f"Confidence: `{confidence:.2f}`")
