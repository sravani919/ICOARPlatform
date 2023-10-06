import extra_streamlit_components as stx
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="ICOAR", layout="wide")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

menu_options = ["Home", "Data Collection", "Text Analysis", "Multimedia Analysis"]
selected = option_menu(
    None,
    menu_options,
    icons=["house-fill", "box-arrow-in-down", "chat-left-text", "images"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px", "--hover-color": "#B9B5B5"},
    },
)
if selected == menu_options[0]:
    from tabs.home_page import home_content

    home_content()

elif selected == menu_options[1]:
    # Data collection
    from tabs.Data_Collection.data_collection_tab import data_collection_tab

    data_collection_tab()

elif selected == menu_options[2]:
    text_sections = ["Text Classification", "Text Annotation"]
    selected_text_section = option_menu(
        None,
        text_sections,
        icons=["chat-left-text", "tag-fill", "clipboard-data-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={},
    )

    if selected_text_section == text_sections[0]:
        text_stepper_options = ["Classification", "Visualisation"]

        selected_option = stx.stepper_bar(steps=text_stepper_options, lock_sequence=False)

        if selected_option == 0:
            from tabs.validation.validation import validation

            st.subheader("Classification")
            st.markdown(":bulb: In this tab, you can use a pretrained model to label datasets depdending on the task.")

            multi = """:bulb: Steps -
                    :one:  Select the dataset that you want to have labeled by using the dropdown.
                    :two:   Choose a model from our list of recommended ones or find a specific one via huggingface
                    :three:  Click on the predict button and view or save your results.
                        """
            st.markdown(multi)
            validation()

        elif selected_option == 1:
            from tabs.Visualisation.Text_Visualisation import Text_Visualisation_tab

            Text_Visualisation_tab()
    # Code above means add visualization into tab "Visualization"

    elif selected_text_section == text_sections[1]:
        from tabs.Text_Annotation.Text_annotation import text_annotation_tab

        text_annotation_tab()

    elif selected_text_section == text_sections[2]:
        from tabs.validation.validation import validation

        st.subheader("Validation")
        st.markdown(":bulb: In this tab, you can use a pretrained model to label datasets depdending on the task.")

        multi = """:bulb: Steps -
                :one:  Select the dataset that you want to have labeled by using the dropdown.
                :two:   Choose a model from our list of recommended ones or find a specific one via huggingface
                :three:  Click on the predict button and view or save your results.
                    """
        st.markdown(multi)
        validation()
    # My own code


elif selected == menu_options[3]:
    image_sections = ["Image Classification", "Meme Classification", "Deepfake Detection"]
    selected_image_section = option_menu(
        None,
        image_sections,
        icons=["cloud-arrow-up-fill", "images", "file-earmark-image-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={},
    )

    if selected_image_section == image_sections[0]:
        image_menu_options = ["Retrieval", "Classification"]

        selected_option = stx.stepper_bar(steps=image_menu_options)

        if selected_option == 1:
            from tabs.image.bully_classifification import bully_classification

            bully_classification()

    elif selected_image_section == image_sections[1]:
        from tabs.image.meme_classification import meme_classification

        meme_classification()

    elif selected_image_section == image_sections[2]:
        from tabs.image.deepfake_detection import df_detection

        df_detection()
