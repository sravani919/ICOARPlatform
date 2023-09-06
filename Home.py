import streamlit as st
from streamlit_option_menu import option_menu

from tabs.image.meme_classification import meme_classification

st.set_page_config(layout="wide")


menu_options = ["Text Classification", "Image Classification"]
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
