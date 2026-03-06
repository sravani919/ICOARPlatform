import os
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Only import what you use (no pipeline -> no face_recognition)
from model.df.classifiers import Meso4


if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False


def colored_text(text, color):
    return f'<h3 style="color:{color}">{text}</h3>'


def pred(generator):
    X, _y = next(generator)  # ✅ use next()
    classifier = Meso4()
    classifier.load("model/df/Meso4_DF.h5")

    p = classifier.predict(X)
    # your original logic: >0.5 means Original
    return "Original image" if p[0][0] > 0.5 else "Deep Fake image"


def df_detection():
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_image is None:
        st.info("Upload an image to run deepfake detection.")
        return

    st.session_state.image_uploaded = True
    save_directory = "data/deepfake/real"
    os.makedirs(save_directory, exist_ok=True)

    img_path = os.path.join(save_directory, "test_image.jpeg")
    with open(img_path, "wb") as f:
        f.write(uploaded_image.read())

    cols = st.columns([0.78, 0.22])
    with cols[1]:
        if st.button("Reset", use_container_width=True):
            if os.path.exists(img_path):
                os.remove(img_path)
            st.session_state.image_uploaded = False
            st.rerun()

    data_gen = ImageDataGenerator(rescale=1.0 / 255)

    generator = data_gen.flow_from_directory(
        "data/deepfake",
        target_size=(256, 256),
        batch_size=1,
        class_mode="binary",
    )

    pred_op = pred(generator)

    if st.session_state.image_uploaded:
        col1, col2, col3 = st.columns(3)
        with col2:
            if pred_op == "Deep Fake image":
                st.markdown(colored_text("Deepfake Image", "red"), unsafe_allow_html=True)
            else:
                st.markdown(colored_text("Original Image", "green"), unsafe_allow_html=True)
            st.image(img_path)