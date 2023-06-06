import glob

import gensim
import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import streamlit as st
from nltk.tokenize import word_tokenize

if "filename_pred" not in st.session_state:
    st.session_state.filename_pred = ""

option = st.selectbox("Select a file", [file for file in glob.glob("./predicted/*.csv")])

if st.button("Load"):
    st.session_state.filename_pred = option

if st.session_state.filename_pred != "":
    df = pd.read_csv(st.session_state.filename_pred)
    options = ["Topic Modeling", "Bar Plot", "Pie Chart"]
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

    elif selected_option == "Topic Modeling":
        data = data["text"].tolist()

        # Tokenize the documents
        tokenized_data = [word_tokenize(text) for text in data]

        # Create dictionary and corpus
        dictionary = gensim.corpora.Dictionary(tokenized_data)
        corpus = [dictionary.doc2bow(text) for text in tokenized_data]

        # Train LDA model
        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

        # Extract topic labels
        topic_labels = []
        topics = lda_model.show_topics(num_topics=10, num_words=10, formatted=False)
        for topic in topics:
            words = [word for word, _ in topic[1]]
            topic_labels.append(" ".join(words))

        # Display topic labels
        for i, label in enumerate(topic_labels):
            st.write(f"Topic {i + 1}: {label}")

        # Prepare data for visualization
        vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
        html_string = pyLDAvis.prepared_data_to_html(vis_data)

        st.components.v1.html(html_string, width=1000, height=600, scrolling=True)
