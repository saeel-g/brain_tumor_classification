import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

with st.sidebar:
    st.markdown('''
    # About
    This Streamlit app classifies brain tumor MRI images into 4 different classes 
    - **Classes:**
        - Glioma: [(Details)](https://en.wikipedia.org/wiki/Glioma)          
        - Meningioma: [(Details)](https://en.wikipedia.org/wiki/Meningioma)  
        - Pituitary:[(Details)](https://en.wikipedia.org/wiki/Pituitary_adenoma) 
        - No Tumor: No tumor is present 
    
    Using the InceptionResNetV2 model.
    \n Made by Saeel Gote [(GitHub)](https://github.com/saeel-g)
    \n Find the code on [GitHub](https://github.com/saeel-g/brain_tumor_classification)
''')

model = load_model('./trained models/train_InceptionResNetV2.h5')

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title('NeuroScan: Classifying Brain Tumor MRI Images')

file = st.file_uploader('Upload an image', type=['.jpg', '.png', '.jpeg'],help="Minimum Image resolution Should be 256x256 px")

if file is not None:
    # Display the uploaded image
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', use_column_width=False)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # st.image(processed_image, caption='processed Image.', use_column_width=False)
    # Make predictions
    predictions = model.predict(processed_image)

    # Display the predicted class
    if np.argmax(predictions)==0:
        st.write(f"Predicted Class: Glioma ")
    elif np.argmax(predictions)==1:
        st.write(f"Predicted Class: Meningioma ")
    elif np.argmax(predictions)==2:
        st.write(f"Predicted Class: No Tumor ")
    elif np.argmax(predictions)==3:
        st.write(f"Predicted Class: Pituitary ")
