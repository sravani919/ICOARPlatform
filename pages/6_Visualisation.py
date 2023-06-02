import glob

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

if "filename_pred" not in st.session_state:
    st.session_state.filename_pred = ""

option = st.selectbox("Select a file", [file for file in glob.glob("./predicted/*.csv")])

if st.button("Load"):
    st.session_state.filename_pred = option

if st.session_state.filename_pred != "":
    df = pd.read_csv(st.session_state.filename_pred)
    options = ["Bar Plot", "Pie Chart"]
    selected_option = st.selectbox("Select an type of visualisation", options)
    data = df
    fig, ax = plt.subplots()

    value_counts = data["sentiment"].value_counts()

    if selected_option == "Bar Plot":
        fig, ax = plt.subplots()
        ax.bar(value_counts.index, value_counts.values)

        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Classification of tweets")

        st.pyplot(fig)
    elif selected_option == "Pie Chart":
        fig, ax = plt.subplots()
        ax.pie(value_counts.values, labels=data["sentiment"].unique(), autopct="%1.1f%%")
        ax.set_title("Pie Chart")
        st.pyplot(fig)
