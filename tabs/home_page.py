import streamlit as st


def home_content():
    st.header("Cyberinfrastructure for Online Abuse Research")
    st.markdown(
        """On this portal, we enable researchers to study and anlyse cyber-bulling.
        We have various functionalities on the portal. They are dicided into 2 main sub-tasks.
        i.e. **Text and Multi-media analysis**.
                """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.header("1. Text Analysis")
        st.markdown(
            """On this tab, we have 3 functionalities - **Text Classification, Text Annotation and Emotion Analysis**
                    """
        )
        st.subheader("i. Text Classification")
        st.subheader("ii. Text Annotation")
        st.subheader("iii. Emotion Analysis")

    with col2:
        st.header("2. Multimedia Analysis")

        st.subheader("i. Image Classification")
        st.subheader("ii. Meme Classification")
        st.subheader("iii. Deepfake Detection")
