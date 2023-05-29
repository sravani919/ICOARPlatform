import re
import string

import gensim
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

if "output" not in st.session_state or st.session_state is None:
    st.warning("Please run the Validation step before the Visualisation step!")
else:
    options = ["Topic Modeling", "Bar Plot", "Pie Chart"]
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

    elif selected_option == "Topic Modeling":
        data["text"] = data["text"].astype(str)

        # Preprocess text
        data["text"] = data["text"].str.lower()  # Convert text to lowercase
        data["text"] = data["text"].apply(lambda x: re.sub("[^A-Za-z]", " ", x))  # Remove special characters
        data["text"] = data["text"].apply(
            lambda x: " ".join([word for word in word_tokenize(str(x)) if word.isalpha()])
        )  # Tokenization

        # Continue with the remaining preprocessing steps
        stop_words = set(stopwords.words("english"))
        data["text"] = data["text"].apply(
            lambda x: " ".join([word for word in word_tokenize(x) if word not in stop_words])
        )  # Remove stopwords
        lemmatizer = WordNetLemmatizer()
        data["text"] = data["text"].apply(
            lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)])
        )  # Lemmatization

        # Preprocess data for LDA
        data = data["text"].tolist()
        preprocessed_data = []
        for text in data:
            tokens = word_tokenize(text)
            filtered_tokens = [
                token for token in tokens if token.lower() not in stop_words and token not in string.punctuation
            ]
            preprocessed_data.append(filtered_tokens)

        # Create dictionary and corpus
        dictionary = gensim.corpora.Dictionary(preprocessed_data)
        corpus = [dictionary.doc2bow(text) for text in preprocessed_data]

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
