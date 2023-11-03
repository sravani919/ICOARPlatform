def login_error():
    import streamlit as st

    st.error("Please login on the home page to view this tab")


def login_success():
    import streamlit as st
    import streamlit.components.v1 as components

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
