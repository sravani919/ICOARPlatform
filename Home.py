import streamlit as st
from streamlit_option_menu import option_menu

from tabs.image.meme_classification import meme_classification
from tabs.Text_Annotation.Text_annotation import text_annotation_tab
from tabs.validation.validation import validation

st.set_page_config(layout="wide")


menu_options = ["Text Classification", "Image Classification", "Validation", "Text Annotation"]
selected = option_menu(
    None,
    menu_options,
    icons=["house", "cloud-upload", "list-task", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
    },
)

if selected == menu_options[0]:
    text_menu_options = ["Retireval", "Preprocessing", "Classification"]

    selected_option = option_menu(
        None,
        text_menu_options,
        icons=["house", "cloud-upload", "list-task", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={"nav-link-selected": {"background-color": "red"}},
    )

    if selected_option == "Retireval":
        st.header("Work in progress")
    elif selected_option == "Preprocessing":
        st.header("Work in progress")
    elif selected_option == "Classification":
        st.header("Work in progress")

if selected == menu_options[1]:
    image_menu_options = ["Retrieval", "Classification", "Meme Classification", "Deepfake Detection"]

    selected_option = option_menu(
        None,
        image_menu_options,
        icons=["house", "cloud-upload", "list-task", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={"nav-link-selected": {"background-color": "red"}},
    )

    if selected_option == "Meme Classification":
        st.markdown(":bulb: In this tab, you can classify memes as ***Hateful*** or ***Non-Hateful***.")

        multi = """:bulb: Steps -
        :one:  Upload a zip file containing memes (jpeg, png, jpg) format.
        :two:  We will extract the text from the memes for you.
        :three:  Make sure it is matching to the actual text in the meme.
        :four:  Click on the predict button.
            """
        st.markdown(multi)

        meme_classification()

if selected == menu_options[2]:
    st.subheader("Validation")
    st.markdown(":bulb: In this tab, you can use a pretrained model to label datasets depdending on the task.")

    multi = """:bulb: Steps -
            :one:  Select the dataset that you want to have labeled by using the dropdown.
            :two:   Choose a model from our list of recommended ones or find a specific one via huggingface
            :three:  Click on the predict button and view or save your results.
                """
    st.markdown(multi)
    validation()


if selected == menu_options[3]:
    # st.subheader("Text Annotation")

    text_annotation_tab()
