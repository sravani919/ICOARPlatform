import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu


def classification_headers(email):
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
        _discrete_slider = components.declare_component("discrete_slider_2", url="http://localhost:3001")

        selected_option = 0
        options = []

        temp = _discrete_slider(options=options, key=None)
        if temp:
            selected_option = temp

        if selected_option == 0:
            from tabs.validation.validation import validation

            st.subheader("Classification")
            st.markdown(":bulb: In this tab, you can use a pretrained model to label datasets depdending on the task.")

            # multi = """:bulb: Steps -
            #         :one:  Select the dataset that you want to have labeled by using the dropdown.
            #         :two:   Choose a model from our list of recommended ones or find a specific one via huggingface
            #         :three:  Click on the predict button and view or save your results.
            #             """
            # st.markdown(multi)
            validation(email)

        elif selected_option == 1:
            from tabs.Visualisation.Text_Visualisation import Text_Visualisation_tab

            Text_Visualisation_tab()
    # Code above means add visualization into tab "Visualization"

    elif selected_text_section == text_sections[1]:
        from tabs.Text_Annotation.Text_annotation import text_annotation_tab

        text_annotation_tab()
