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
            _discrete_slider = components.declare_component("discrete_slider", url="http://localhost:3001")

            activeStep = 0
            options = []
            _discrete_slider(options=options, key=None, default=0)

            selected_option = activeStep

            if selected_option == 0:
                from tabs.validation.validation import validation

                validation()

            elif selected_option == 1:
                from tabs.Visualisation.Text_Visualisation import Text_Visualisation_tab

                Text_Visualisation_tab()
        # Code above means add visualization into tab "Visualization"

        elif selected_text_section == text_sections[1]:
            from tabs.Text_Annotation.Text_annotation import text_annotation_tab

            text_annotation_tab()

    elif selected_value == 5:
        st.success("You're logged out sucessfuly. Please refresh the page.")
