import os

import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from tabs.login import login_error

if "authenticator" not in st.session_state:
    st.session_state.authenticator = None
if "user_login" not in st.session_state:
    st.session_state.user_login = True
if "user_registration" not in st.session_state:
    st.session_state.user_registration = False
if "user_registration_complete" not in st.session_state:
    st.session_state.user_registration_complete = False

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

production = True

if production:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(root_dir, "header_tab/build")

    _discrete_slider = components.declare_component("discrete_slider", path=build_dir)
else:
    _discrete_slider = components.declare_component("discrete_slider", url="http://localhost:3000")


def discrete_slider():
    return _discrete_slider(default=0, logged_in=False)


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
                    margin-top: {1}%;
                }}
            </style>
            """,
    unsafe_allow_html=True,
)
selected_value = int(discrete_slider())

if selected_value == 0:
    from tabs.login import login

    with open(".streamlit/authenticator.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)
    st.session_state.authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["preauthorized"],
    )

    login(st.session_state.authenticator, config)

elif selected_value == 1:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        from tabs.Data_Collection.data_collection_tab import data_collection_tab

        data_collection_tab()

elif selected_value == 2:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        cols = st.columns(1)
        with cols[0]:
            from tabs.Data_Collection.data_preprocessing_tab import data_preprocessing_tab

            data_preprocessing_tab()

elif selected_value == 3:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        from tabs.validation.validation import validation

        cols = st.columns(1)
        with cols[0]:
            validation()

elif selected_value == 4:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        from tabs.Text_Annotation.Text_annotation import text_annotation_tab

        cols = st.columns(1)
        with cols[0]:
            text_annotation_tab()

elif selected_value == 5:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        from tabs.Visualisation.Text_Visualisation import Text_Visualisation_tab

        cols = st.columns(1)
        with cols[0]:
            Text_Visualisation_tab()

elif selected_value == 6:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        cols = st.columns(1)
        with cols[0]:
            image_sections = ["Image Analysis", "Meme Analysis", "Deepfake Detection"]
            selected_image_section = selected_data = st.selectbox("Select type of multi-media analysis", image_sections)
            st.markdown("-------------------")
            st.subheader(selected_image_section)
            if selected_image_section == image_sections[0]:
                from tabs.image.bully_classifification import bully_classification

                bully_classification()

            elif selected_image_section == image_sections[1]:
                from tabs.image.meme_classification import meme_classification

                meme_classification()

            elif selected_image_section == image_sections[2]:
                from tabs.image.deepfake_detection import df_detection

                df_detection()
elif selected_value == 8:
    if "authentication_status" not in st.session_state or not st.session_state["authentication_status"]:
        st.warning("You're logged out. Please sign in to access the features")
    else:
        cols = st.columns(1)

        with cols[0]:
            st.subheader("Account Details")
            st.markdown("**Name**: " + st.session_state["name"])
            st.markdown("**Username**: " + st.session_state["username"])
            st.session_state.authenticator.logout("Logout", "main", key="unique_key")
