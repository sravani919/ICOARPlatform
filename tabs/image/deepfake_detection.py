import streamlit as st
import numpy as np
from model.df.classifiers import *
from model.df.pipeline import *
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

def colored_text(text, color):
    return f'<h3 style="color:{color}">{text}</h3>'

def pred(generator):
    X, y = generator.next()
    op = 'Deep Fake image'
    classifier = Meso4()
    classifier.load('model/df/Meso4_DF.h5')
    pred = classifier.predict(X)
    if pred[0][0] > 0.5:
        op = 'Original image'

    return op

def df_detection():
    uploaded_image = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_image is not None:
        st.session_state.image_uploaded = True
        save_directory = "data/deepfake/real"
        os.makedirs(save_directory, exist_ok=True)
        
        with open(os.path.join(save_directory, "test_image.jpeg"), "wb") as f:
            f.write(uploaded_image.read())
        

        if st.button("Reset"):
            os.remove(save_directory + "/test_image.jpeg")
            st.session_state.image_uploaded = False
            uploaded_image = None

        dataGenerator = ImageDataGenerator(rescale=1./255)
        generator = dataGenerator.flow_from_directory(
                "data/deepfake",
                target_size=(256, 256),
                batch_size=1,
                class_mode='binary',
                subset='training')

        pred_op = pred(generator)

        if st.session_state.image_uploaded:
            col1, col2, col3 = st.columns(3)
            with col2:
                if pred_op == "Deep Fake image":
                    st.markdown(colored_text("Deepfake Image", "red"), unsafe_allow_html=True)
                else:
                    st.markdown(colored_text("Orginal Image", "red"), unsafe_allow_html=True)
                st.image(save_directory + "/test_image.jpeg")