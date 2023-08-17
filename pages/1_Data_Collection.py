import streamlit as st

title = "Data Collection"

st.set_page_config(page_title=title)

st.sidebar.header(title)

option = st.sidebar.selectbox(
    "Choose a source",
    [
        "Twitter (API)",
        "Twitter (Scraper)",
        "Reddit (Scraper)",
        "Tiktok (API)",
        "Tiktok (Scraper)",
        "Facebook (Scraper)",
    ],
)
thing_names = {"Twitter": "tweets", "Reddit": "posts", "Tiktok": "tiktoks", "Facebook": "posts"}
