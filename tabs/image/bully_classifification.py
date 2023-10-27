import glob as glob
import os
import zipfile

import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras

DATA_DIR = "././data/images/image/"


def preprocess_image(image):
    # Decode and resize image.
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    return image


def load_and_preprocess_image(path):
    # Read image into memory as a byte string.
    image = tf.io.read_file(path)
    return preprocess_image(image)


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


# delete all files in DATA_DIR
def empty_cache():
    file_list = os.listdir(DATA_DIR)
    for file_name in file_list:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    st.session_state.image_uploaded = False


def bully_classification():
    if "image_uploaded" not in st.session_state:
        st.session_state.image_uploaded = False

    file_ = st.file_uploader(
        "Upload an image file, or zipfile. Make sure the extension of images is either jpg or png.",
        type=["jpg", "png", "zip"],
    )

    # if st.button("Reset"):
    #     empty_cache()

    if file_ is not None:
        # if DATA_DIR is not empty, reset()
        if len(os.listdir(DATA_DIR)) > 0:
            empty_cache()
        # if file_ is a zipfile, extract it
        if file_.type == "application/zip":
            save_path = os.path.join(DATA_DIR, "image.zip")
            with open(save_path, "wb") as f:
                f.write(file_.getbuffer())
            with zipfile.ZipFile(save_path, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)
            os.remove(save_path)
            st.success("File saved successfully!")
            st.session_state.image_uploaded = True
        # if file_ is an image, save it
        else:
            file_path = os.path.join(DATA_DIR, file_.name)
            with open(file_path, "wb") as f:
                f.write(file_.read())
            st.success("File saved successfully!")
            st.session_state.image_uploaded = True

    # make predictions on the saved images
    if st.session_state.image_uploaded:
        st.success("Predicting... Please wait...")
        model = keras.models.load_model("model/fine_tuned_vgg16_model.h5")

        image_paths = sorted(glob.glob("data/images/image" + "/*.jpg") + glob.glob("data/images/image" + "/*.png"))

        label_ids = [0] * len(image_paths)

        test_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_ids))

        test_dataset = test_dataset.map(load_and_preprocess_from_path_label)

        test_dataset = test_dataset.batch(16)

        class_names = ["cyberbullying", "non_cyberbullying"]

        # Evaluate all the batches.
        st.subheader("Predictions")
        for image_batch, labels_batch in test_dataset.take(1):
            # Predictions for the current batch.
            predictions = model.predict(image_batch)

            # Loop over all the images in the current batch.
            for idx in range(len(labels_batch)):
                pred_idx = tf.argmax(predictions[idx]).numpy()

                title = str(class_names[pred_idx])

                if title == "cyberbullying":
                    color_style = "color: red;"
                else:
                    color_style = "color: green;"

                st.write(f'<h3 style="{color_style}">{title}</h3>', unsafe_allow_html=True)
                image = Image.open(image_paths[idx])
                resized_image = image.resize((200, 200))
                st.image(resized_image, use_column_width=False)
