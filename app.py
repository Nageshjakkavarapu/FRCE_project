import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
@st.cache(allow_output_mutation=True)
def load_keras_model():
    return load_model("keras_Model.h5", compile=False)

model = load_keras_model()

# Load the labels
@st.cache
def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f]

class_names = load_labels()

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array.reshape(1, *normalized_image_array.shape)

# Streamlit UI
st.title("Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence_score = prediction[0][predicted_class_index]

    # Display prediction
    st.write("Prediction: ", predicted_class)
    st.write("calories=C,iron=I,fat=F,carbohidrates-CB,vitomins=V")
    st.write("Confidence Score: ", confidence_score)
