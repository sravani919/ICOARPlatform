# from common.meme.classification import classify_memes
import json
import os

import streamlit as st

print("import complete")

if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False
if "caption_uploaded" not in st.session_state:
    st.session_state.caption_uploaded = False
if "predict" not in st.session_state:
    st.session_state.predict = False

uploaded_meme = st.file_uploader("Upload a meme", type=["jpg", "jpeg", "png"])
if uploaded_meme and not st.session_state.image_uploaded:
    print("meme - ", uploaded_meme)
    uploaded_meme.name = "1"
    with open(os.path.join("data/img", "1.png"), "wb") as f:
        f.write(uploaded_meme.getbuffer())
    st.session_state.image_uploaded = True

caption = st.text_input("Enter the caption here")

if len(caption):
    file_path = "data/test_seen2.jsonl"
    with open(file_path, "r") as file:
        data = json.load(file)
    data["text"] = caption
    with open(file_path, "w") as file:
        json.dump(data, file)
    st.session_state.caption_uploaded = True
if st.button("Predict"):
    st.session_state.predict = True


def reset():
    st.session_state.predict = False
    st.session_state.caption_uploaded = False
    st.session_state.image_uploaded = False
    st.session_state.files_uploaded = False


if st.button("Reset"):
    reset()

if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False

if st.session_state.image_uploaded and st.session_state.caption_uploaded and st.session_state.predict:
    st.session_state.files_uploaded = True

if st.session_state.files_uploaded:
    with st.spinner("Task is in progress..."):
        from common.meme.classification import classify_memes

        print("Result - ", classify_memes())
        result = classify_memes()
        pred = result["preds"][0]

        title = "Hateful meme"
        if pred:
            color_style = "color: red;"
        else:
            color_style = "color: green;"
            title = "Non-hateful meme"

        st.write(f'<h3 style="{color_style}">{title}</h3>', unsafe_allow_html=True)
        reset()
    print("Completed")
