import streamlit as st

from tabs.image.meme_classification import meme_classification
from tabs.Text_Annotation.Text_annotation import text_annotation_tab
from tabs.validation.validation import validation

st.set_page_config(layout="wide")


def read_css_file():
    with open("style.css") as f:
        return f.read()


st.markdown(f"<style>{read_css_file()}</style>", unsafe_allow_html=True)


tab1, tab2, validation_tab, annotation_tab = st.tabs(
    ["Text Classification", "Image Classification", "Validation", "Text Annotation"]
)

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

with validation_tab:
    st.subheader("Validation")
    st.markdown(":bulb: In this tab, you can use a pretrained model to label datasets depdending on the task.")

    multi = """:bulb: Steps -
            :one:  Select the dataset that you want to have labeled by using the dropdown.
            :two:   Choose a model from our list of recommended ones or find a specific one via huggingface
            :three:  Click on the predict button and view or save your results.
                """
    st.markdown(multi)
    validation()


with annotation_tab:
    # st.subheader("Text Annotation")

    text_annotation_tab()
