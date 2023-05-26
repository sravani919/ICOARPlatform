import matplotlib.pyplot as plt
import streamlit as st

if "output" not in st.session_state or st.session_state.output.empty:
    st.warning("Please run the Validation step before the Visualisation step!")
else:
    options = ["Bar Plot", "Pie Chart"]
    selected_option = st.selectbox("Select an type of visualisation", options)
    data = st.session_state.output
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
