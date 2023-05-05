import glob

import pandas as pd
import streamlit as st

from data_preprocessing import preprocess

title = "Data Preprocessing"

if "filename" not in st.session_state:
    st.session_state.filename = ""
if "preprocessing_status" not in st.session_state:
    st.session_state.preprocessing_status = False

st.set_page_config(page_title=title)

st.sidebar.header(title)
option = st.sidebar.selectbox("Select a file", [file for file in glob.glob("./data/*.csv")])

if st.sidebar.button("Load"):
    st.session_state.filename = option

if st.session_state.filename != "":
    df = pd.read_csv(st.session_state.filename)
    st.dataframe(df)
    english_only = st.checkbox("Filter Out Non-English Tweets", value=False)
    tweet_clean = st.checkbox("Remove URLs, Hashtags, Mentions, Emojis", value=False)
    if st.button("Process"):
        st.session_state.preprocessing_status = preprocess(st.session_state.filename, english_only, tweet_clean)

if st.session_state.preprocessing_status:
    df = pd.read_csv(st.session_state.filename)
    st.dataframe(df)
