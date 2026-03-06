import os
import streamlit as st
import streamlit.components.v1 as components
import yaml


# -------------------------------------------------
# Components
# -------------------------------------------------
def _corousel_component():
    production = True
    if production:
        build_dir = "./corousel/build"
        return components.declare_component("corousel", path=build_dir)
    return components.declare_component("corousel", url="http://localhost:3000")


def _citations_component():
    production = True
    if production:
        build_dir = "./citations/build"
        return components.declare_component("citations", path=build_dir)
    return components.declare_component("citations", url="http://localhost:3001")


def corousel_info():
    """
    Legacy layout: 2 columns (carousel left, citations right)
    Keeping this so nothing breaks if old code calls it.
    """
    cols = st.columns(2)
    with cols[0]:
        _corousel_component()()
    with cols[1]:
        _citations_component()()


def carousel_only(height=280, key="home_corousel"):
    production = True
    if production:
        build_dir = "./corousel/build"
        corousel = components.declare_component("corousel", path=build_dir)
    else:
        corousel = components.declare_component("corousel", url="http://localhost:3000")
    corousel(height=height, key=key)


def helpful_links_only(title: str = "Helpful Links"):
    """
    Full-width Helpful Links section (citations component).
    Note: The 'About' content you see is likely inside the citations React component.
    """
    if title:
        st.markdown(f"### {title}")
    _citations_component()()


# -------------------------------------------------
# Auth helpers
# -------------------------------------------------
def login_error():
    st.error("Please login on the home page to view this tab")


def user_login(authenticator, config):
    if st.session_state.get("user_registration_complete", False):
        st.success("User registered successfully. Please login")

    authenticator.login("Login", "main")

    if st.session_state.get("authentication_status") is True:
        st.session_state.user_registration_complete = False
    elif st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect")
        st.session_state.user_registration_complete = False
    elif st.session_state.get("authentication_status") is None:
        st.warning("Please enter your username and password")


def user_registration(authenticator, config):
    try:
        if authenticator.register_user("Register user", preauthorization=False):
            with open(".streamlit/authenticator.yaml", "w") as file:
                yaml.dump(config, file, default_flow_style=False)

            st.session_state.user_registration_complete = True
            st.session_state.user_registration = False
            st.session_state.user_login = True
            st.success("User registered successfully. Please login")
    except Exception as e:
        st.error(e)


def create_dir(directory):
    if directory and (not os.path.exists(directory)):
        os.makedirs(directory, exist_ok=True)


# -------------------------------------------------
# Main login UI
# -------------------------------------------------
def login(authenticator, config, parent=None, show_carousel=True, show_success=True, compact=False):
    """
    Login UI with optional carousel + optional success banner.

    - show_carousel=True  -> renders legacy 2-col carousel+citations (corousel_info()).
    - show_success=True   -> renders success banner when logged in.
    - compact=True        -> shows ONLY the login/register form (no extra toggle row)

    For your new Home.py inline hero login:
      - call login(..., show_carousel=False, show_success=False, compact=True)
    """
    p = parent if parent is not None else st

    # Defaults
    if "user_login" not in st.session_state:
        st.session_state.user_login = True
    if "user_registration" not in st.session_state:
        st.session_state.user_registration = False
    if "user_registration_complete" not in st.session_state:
        st.session_state.user_registration_complete = False

    # Only show login/register UI when logged out
    if not st.session_state.get("authentication_status", False):

        # ✅ COMPACT MODE: no "User Login / User Registration" row
        if compact:
            # If you want to force only login (no registration) in hero:
            # st.session_state.user_registration = False
            # st.session_state.user_login = True

            if st.session_state.get("user_registration", False):
                user_registration(authenticator, config)
            else:
                user_login(authenticator, config)

            if st.session_state.get("username"):
                create_dir(f'./data/{st.session_state["username"]}')

        # ✅ FULL MODE: your original behavior (kept)
        else:
            p.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = p.columns([1, 1.2, 0.2, 1.2, 1])

            with c2:
                if p.button("User Login", use_container_width=True, key="btn_user_login"):
                    st.session_state.user_registration = False
                    st.session_state.user_login = True

            with c4:
                if p.button("User Registration", use_container_width=True, key="btn_user_reg"):
                    st.session_state.user_registration = True
                    st.session_state.user_login = False

            if st.session_state.get("user_login", True):
                user_login(authenticator, config)
                if st.session_state.get("username"):
                    create_dir(f'./data/{st.session_state["username"]}')
            else:
                user_registration(authenticator, config)

    # Success banner (optional)
    if show_success and st.session_state.get("authentication_status", False):
        p.success("You're logged in successfully. Use the menu to access the features.")

    # Legacy carousel block (optional)
    if show_carousel:
        corousel_info()
