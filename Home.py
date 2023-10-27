import os

import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import yaml
from streamlit_option_menu import option_menu
from yaml.loader import SafeLoader

st.set_page_config(
    page_title="ICOAR",
    layout="wide",
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

_RELEASE = False

if _RELEASE:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(root_dir, "frontend/build")

    _discrete_slider = components.declare_component("discrete_slider", path=build_dir)
else:
    _discrete_slider = components.declare_component("discrete_slider", url="http://localhost:3000")


def discrete_slider():
    return _discrete_slider(default=0, logged_in=False)


if not _RELEASE:
    st.markdown(
        f"""
                <style>
                    .block-container {{
                        padding: {0}rem;
                    }}
                    header[data-testid="stHeader"] {{
                        display: none;
                    }}
                    div[data-testid="stVerticalBlock"] {{
                        gap: {0}rem
                    }}
                    div[data-testid="stForm"]  {{
                        margin-left: {5}%;
                        margin-right: {5}%;
                        margin-top: {1.5}%;
                    }}
                    div[data-testid="stFormSubmitButton"]  {{
                        margin-top: {1.5}%;
                    }}
                    .stAlert {{
                        margin-left: {5}%;
                        margin-right: {5}%;
                        margin-top: {1.5}%;
                    }}
                    div[data-testid="stHorizontalBlock"] {{
                        margin-left: {7.5}%;
                        margin-right: {7.5}%;
                        margin-top: {2.5}%;
                    }}
                </style>
                """,
        unsafe_allow_html=True,
    )
    selected_value = int(discrete_slider())

    if selected_value == 0:
        with open(".streamlit/authenticator.yaml") as file:
            config = yaml.load(file, Loader=SafeLoader)

        authenticator = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
            config["preauthorized"],
        )

        authenticator.login("Login", "main")

        if st.session_state["authentication_status"]:
            st.success("You're logged in sucessfuly. Use the menu bar to access the features.")
        elif st.session_state["authentication_status"] is False:
            st.error("Username/password is incorrect")
        elif st.session_state["authentication_status"] is None:
            st.warning("Please enter your username and password")

    elif selected_value == 1:
        from tabs.Data_Collection.data_collection_tab import data_collection_tab

        data_collection_tab()

    elif selected_value == 2:
        from tabs.validation.validation import validation

        validation()

    elif selected_value == 3:
        from tabs.Text_Annotation.Text_annotation import text_annotation_tab

        text_annotation_tab()

    elif selected_value == 4:
        from tabs.Visualisation.Text_Visualisation import Text_Visualisation_tab

        Text_Visualisation_tab()

    elif selected_value == 5:
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
            selected_option = 0
            if selected_option == 0:
                pass

            if selected_option == 1:
                from tabs.image.bully_classifification import bully_classification

                bully_classification()

        elif selected_image_section == image_sections[1]:
            from tabs.image.meme_classification import meme_classification

            meme_classification()

        elif selected_image_section == image_sections[2]:
            from tabs.image.deepfake_detection import df_detection

            df_detection()
    elif selected_value == 7:
        st.success("You're logged out sucessfuly. Please refresh the page.")
