import streamlit as st


def home_content():
    st.header("Cyberinfrastructure for Online Abuse Research")
    st.markdown(
        """ On this portal, we enable researchers to study and anlyse cyber-bulling.
        We have various functionalities on the portal. They are dicided into 2 main sub-tasks.
        i.e. **Text and Multi-media analysis**.
                """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.header("1. Text Analysis")
        st.markdown(
            """On this tab, we have 2 functionalities - **Text Classification, Text Annotation**
                    """
        )
        st.subheader("i. Text Classification")
        st.markdown(
            """In this tab you can classify data extracted from social media into hateful and non-hateful content.
                You can chose from a variety of models available online to do your analysis. You can even use your
                own custom models.
                """
        )
        st.subheader("ii. Text Annotation")

    with col2:
        st.header("2. Multimedia Analysis")
        st.markdown(
            """In this tab you can analyse multi-media data.
            Various multi-media classification functionalities are provided in this tab."""
        )
        st.subheader("i. Image Classification")
        st.markdown(
            """A custom model has developed to categorise images to detect hateful content.
            The model can be used for image classification."""
        )
        st.subheader("ii. Meme Classification")
        st.markdown(
            """Classifying memes is a slightly difficult process, since changing either
              the caption or the text in the meme can change the context.
                     Hence, we have used a multi-model model to detect hateful content in memes."""
        )
        st.subheader("iii. Deepfake Detection")
        st.markdown("""Using this functionality you can detect fake images or AI generated images.""")