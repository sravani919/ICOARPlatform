# from common.meme.classification import classify_memes
import os
import ssl

import streamlit as st

from common.meme.utility import (
    clear_files,
    data_folder,
    extract_file,
    generate_json,
    get_caption,
    memes_folder_name,
    verify_files,
)

# Added to avoid the ssl certificate expiry error
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(layout="wide")

# Init the states
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False
if "caption_uploaded" not in st.session_state:
    st.session_state.caption_uploaded = False
if "predict" not in st.session_state:
    st.session_state.predict = "False"
if "file_list" not in st.session_state:
    st.session_state.file_list = []
if "caption_list" not in st.session_state:
    st.session_state.caption_list = []
if "pred_list" not in st.session_state:
    st.session_state.pred_list = []
if "generate_caption" not in st.session_state:
    st.session_state.generate_caption = False


def reset():
    clear_files()
    st.session_state.predict = "False"
    st.session_state.image_uploaded = False
    st.session_state.caption_uploaded = False
    st.session_state.predict = "False"
    st.session_state.file_list = []
    st.session_state.caption_list = []
    st.session_state.pred_list = []
    st.session_state.generate_caption = False


uploaded_meme = st.file_uploader("Upload a meme", type=["zip"])


# Unzip the file and verify if it contains images
if uploaded_meme and not st.session_state.image_uploaded:
    uploaded_meme.name = "memes"
    with open(os.path.join("data/", "memes.zip"), "wb") as f:
        f.write(uploaded_meme.getbuffer())
    extract_file()
    file_list, verified = verify_files()
    st.session_state.file_list = file_list

    if verified:
        st.success("Upload Successful")
        st.session_state.image_uploaded = True
    else:
        st.error("The folder contains non-image file extensions. We support only jpg, jpeg, png")


if not st.session_state.generate_caption and st.session_state.image_uploaded:
    progress_text = "Extracting text from the images..."
    progress_bar = st.progress(0, text=progress_text)
    total_img = len(st.session_state.file_list)

    for index, image in enumerate(st.session_state.file_list):
        progress_bar.progress(int(((index + 1) / total_img) * 100), text=progress_text)
        caption = get_caption(data_folder + memes_folder_name + "/" + image)
        st.session_state.caption_list.append(caption)
    st.session_state.generate_caption = True


def on_caption_change(index, placeholder):
    st.session_state.caption_list[index] = st.session_state[index]


if st.session_state.generate_caption:
    with st.container():
        col = st.columns(6)
        with col[3]:
            if st.button("Reset", type="primary"):
                reset()
        with col[2]:
            if st.session_state.predict != "Completed":
                if st.button("Predict"):
                    st.session_state.predict = "True"

    i = 0
    if st.session_state.predict == "False":
        while i < len(st.session_state.file_list):
            col = st.columns(3)
            j = 0
            with st.container():
                while j < 3 and i < len(st.session_state.file_list):
                    image = st.session_state.file_list[i]
                    with col[j]:
                        st.image(data_folder + memes_folder_name + "/" + image)
                        caption = st.session_state.caption_list[i]
                        st.text_input(
                            "Caption (Please verify): ",
                            value=caption,
                            key=i,
                            on_change=on_caption_change,
                            args=(i, "placeholder"),
                        )
                        i += 1
                    j += 1

generate_json(st.session_state.file_list, st.session_state.caption_list)

if st.session_state.predict == "True":
    with st.spinner("Prediction is in progress..."):
        from common.meme.classification import classify_memes

        result = classify_memes()
        pred = result["preds"][0]

        st.session_state.pred_list = pred = result["preds"]

        st.session_state.predict = "Completed"

if st.session_state.predict == "Completed":
    i = 0
    while i < len(st.session_state.file_list):
        col = st.columns(3)
        j = 0
        with st.container():
            while j < 3 and i < len(st.session_state.file_list):
                image = st.session_state.file_list[i]
                with col[j]:
                    title = "Hateful meme"
                    if st.session_state.pred_list[i]:
                        color_style = "color: red;"
                    else:
                        color_style = "color: green;"
                        title = "Non-hateful meme"

                    st.write(f'<h3 style="{color_style}">{title}</h3>', unsafe_allow_html=True)
                    st.image(data_folder + memes_folder_name + "/" + image)

                    i += 1
                j += 1
