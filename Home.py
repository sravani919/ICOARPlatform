import streamlit as st

from tabs.image.meme_classification import meme_classification

st.set_page_config(layout="wide")


def read_css_file():
    with open("style.css") as f:
        return f.read()


st.markdown(f"<style>{read_css_file()}</style>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Text Classification", "Image Classification"])

with tab1:
    st.subheader("Text classification")
    tab1_1, tab1_2, tab1_3 = st.tabs(["Retireval", "Preprocessing", "Classification"])
    with tab1_1:
        st.subheader("Work in progress")

with tab2:
    st.subheader("Image classification")
    tab2_1, tab2_2, tab2_3, tab2_4 = st.tabs(
        ["Retrieval", "Classification", "Meme Classification", "Deepfake Detection"]
    )
    with tab2_1:
        st.subheader("Work in progress")
    with tab2_3:
        st.markdown(":bulb: In this tab, you can classify memes as ***Hateful*** or ***Non-Hateful***.")

        multi = """:bulb: Steps -
        :one:  Upload a zip file containing memes (jpeg, png, jpg) format.
        :two:  We will extract the text from the memes for you.
        :three:  Make sure it is matching to the actual text in the meme.
        :four:  Click on the predict button.
            """
        st.markdown(multi)

        meme_classification()
