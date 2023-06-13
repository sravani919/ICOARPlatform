import glob

import gensim
import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import streamlit as st
import seaborn as sns
from nltk.tokenize import word_tokenize

if "filename_pred" not in st.session_state:
    st.session_state.filename_pred = ""

option = st.selectbox("Select a file", [file for file in glob.glob("./predicted/*.csv")])

if st.button("Load"):
    st.session_state.filename_pred = option

if st.session_state.filename_pred != "":
    df = pd.read_csv(st.session_state.filename_pred)
    options = ["Topic Modeling","Temporal Analysis", "Bar Plot", "Pie Chart"]
    selected_option = st.selectbox("Select an type of visualisation", options)
    data = df
    fig, ax = plt.subplots()

    

    if selected_option == "Bar Plot":
        fig, ax = plt.subplots()
        value_counts = data["sentiment"].value_counts()
        ax.bar(value_counts.index, value_counts.values)

        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Classification of tweets")

        st.pyplot(fig)
    elif selected_option == "Pie Chart":
        fig, ax = plt.subplots()
        value_counts = data["sentiment"].value_counts()
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

    elif selected_option == "Temporal Analysis":
        fig, ax = plt.subplots(figsize=(10, 6))
        data['date'] = pd.to_datetime(data['date'], errors='coerce')

        data = data[data['date'].notna()]

        keywords = st.text_input("Enter the keywords of interest seperated by comma (i.e., covid, lockdown, ... ):").lower().split(',')
        




        # keywords = input("Enter the keywords of interest seperated by comma (i.e., covid, lockdown, ... ): ")

        masks = [data['text'].str.contains(keyword.strip(), case=False, na=False) for keyword in keywords]

        daily_counts = [data.loc[mask, 'date'].value_counts().sort_index().resample('D').sum() for mask in masks]

        colors = ['blue', 'red', 'purple']
        labels = [f"{keyword.capitalize()}" for keyword in keywords]

        sns.set(style="whitegrid")
        color_palette = sns.color_palette("Set2", len(daily_counts))

        for i, count in enumerate(daily_counts):
            ax.plot(count.index, count.values, color=color_palette[i],linewidth=3, label=labels[i])

        ax.set_xlabel('Date')
        ax.set_ylabel('Tweets Count')
        ax.legend()
        ax.set_xticks(count.index)  # Set the tick locations to count.index
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.grid(True)
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()

        # Display the plot
        st.pyplot(fig)
