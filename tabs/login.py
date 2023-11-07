import streamlit as st
import streamlit.components.v1 as components
import yaml


def corousel_info():
    cols = st.columns(2)
    production = True
    with cols[1]:
        if production:
            build_dir = "./citations/build"

            citations = components.declare_component("citations", path=build_dir)
        else:
            citations = components.declare_component("citations", url="http://localhost:3001")
        citations()

    with cols[0]:
        if production:
            build_dir = "./corousel/build"
            corousel = components.declare_component("corousel", path=build_dir)
        else:
            corousel = components.declare_component("corousel", url="http://localhost:3000")
        corousel()


def login_error():
    st.error("Please login on the home page to view this tab")


def user_login(authenticator, config):
    if st.session_state.user_registration_complete:
        st.success("User registered successfully. Please login")

    authenticator.login("Login", "main")

    if st.session_state["authentication_status"]:
        st.session_state.user_registration_complete = False
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
        st.session_state.user_registration_complete = False
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")


def user_registration(authenticator, config):
    try:
        if authenticator.register_user("Register user", preauthorization=False):
            with open(".streamlit/authenticator.yaml", "w") as file:
                yaml.dump(config, file, default_flow_style=False)
                st.session_state.user_registration_complete = True
                st.session_state.user_registration = False
                st.session_state.user_login = True
            if st.session_state.user_registration_complete:
                st.success("User registered successfully. Please login")
    except Exception as e:
        st.error(e)


def login(authenticator, config):
    if not st.session_state["authentication_status"]:
        cols = st.columns(3)
        with cols[1]:
            inner_cols = st.columns(2)
            with inner_cols[0]:
                if st.button("User Login"):
                    st.session_state.user_registration = False
                    st.session_state.user_login = True
            with inner_cols[1]:
                if st.button("User Registration"):
                    st.session_state.user_registration = True
                    st.session_state.user_login = False

        if st.session_state.user_login:
            user_login(authenticator, config)
        elif st.session_state.user_registration:
            user_registration(authenticator, config)

    if st.session_state["authentication_status"]:
        st.markdown(
            f"""
                <style>
                    .stAlert {{
                        margin-left: 25%;
                        margin-right: 25%;
                        margin-top: -1.5%;
                        height: 50px
                    }}
                    div[data-testid="stHorizontalBlock"] {{
                        margin-left: {7.5}%;
                        margin-right: {7.5}%;
                    }}
                </style>
                """,
            unsafe_allow_html=True,
        )

        st.success("You're logged in successfully. Use the top menu bar to access the features.")

    corousel_info()
