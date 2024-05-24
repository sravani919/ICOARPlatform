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


def selection_bar_1():
    """Declare and call the custom Streamlit component for selection."""
    production = True  # This should be dynamically set based on your environment
    if production:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(root_dir, "header_tab2/build")
        # Ensure this component name is unique in the frontend build
        _selection_bar = components.declare_component("discrete_slider", path=build_dir)
    else:
        _selection_bar = components.declare_component("discrete_slider", url="http://localhost:3000")

    return _selection_bar()


def selection_bar_2():
    """Declare and call the custom Streamlit component for selection."""
    production = True  # This should be dynamically set based on your environment
    if production:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(root_dir, "header_tab3/build")
        # Ensure this component name is unique in the frontend build
        _selection_bar = components.declare_component("discrete_slider", path=build_dir)
    else:
        _selection_bar = components.declare_component("discrete_slider", url="http://localhost:3000")

    return _selection_bar()


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

# Clear the previous component if necessary


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
        from tabs.Visualisation.Text_Visualisation import Text_Visualisation_tab

        cols = st.columns(1)
        with cols[0]:
            Text_Visualisation_tab()

elif selected_value == 5:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        placeholder = st.empty()
        with placeholder.container():
            user_choice_2 = selection_bar_2()
            if user_choice_2 == "Cyberbullying Image Analysis":
                cols = st.columns(1)
                with cols[0]:
                    from tabs.image.bully_classifification import bully_classification

                    bully_classification()
            elif user_choice_2 == "Meme Analysis":
                cols = st.columns(1)
                with cols[0]:
                    from tabs.image.meme_classification import meme_classification

                    meme_classification()
            elif user_choice_2 == "Deepfake Detection":
                cols = st.columns(1)
                with cols[0]:
                    from tabs.image.deepfake_detection import df_detection

                    df_detection()
            elif user_choice_2 == "Customized Image Analysis":
                cols = st.columns(1)
                with cols[0]:
                    from tabs.image.huggingface_image_analysis import huggingface_image_analysis

                    print("Fetching model list from hugging face...")
                    huggingface_image_analysis()
            elif user_choice_2 == "Cyberbullying Detection using GPT":
                cols = st.columns(1)
                with cols[0]:
                    from tabs.image.bully_classifification import image_classification_llm

                    image_classification_llm()


elif selected_value == 6:
    if not st.session_state["authentication_status"]:
        login_error()
    else:
        user_choice = selection_bar_1()

        if user_choice == "Text Annotaion":
            from tabs.Text_Annotation.Text_annotation import text_annotation_tab

            cols = st.columns(1)
            with cols[0]:
                text_annotation_tab(labeling_mode="Text Labeling")

        elif user_choice == "Image Labeling":
            print("labeling")

            from tabs.Text_Annotation.Text_annotation import text_annotation_tab

            cols = st.columns(1)
            with cols[0]:
                text_annotation_tab(labeling_mode="Image Labeling")

        elif user_choice == "Prompt Optimization":
            from tabs.Prompt_Engineering import generate_prompt

            generate_prompt()()

        elif user_choice == "In-Context Learning":
            from tabs.Text_Annotation.In_context_leanring import in_context_learning

            cols = st.columns(1)
            with cols[0]:
                in_context_learning()


elif selected_value == 7:
    if "authentication_status" not in st.session_state or not st.session_state["authentication_status"]:
        st.warning("You're logged out. Please sign in to access the features")
    else:
        cols = st.columns(1)

        with cols[0]:
            st.subheader("Account Details")
            st.markdown("**Name**: " + st.session_state["name"])
            st.markdown("**Username**: " + st.session_state["username"])
            st.session_state.authenticator.logout("Logout", "main", key="unique_key")
