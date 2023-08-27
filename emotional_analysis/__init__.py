from collections import Counter

import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords
from wordcloud import WordCloud


def emotional_analysis(df):
    categories = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]

    freq = [len(st.session_state.output[st.session_state.output["emotion"] == emotion]) for emotion in categories]

    fig1 = go.Figure(data=[go.Pie(labels=categories, values=freq, textinfo="none")])
    fig1.update_layout(title="Frequency Distribution", height=500, width=700)
    st.session_state.pie_chart = fig1

    st.plotly_chart(st.session_state.pie_chart)
    words_array = st.session_state.output["emotion"].unique()
    selected_option = st.selectbox("Select an option:", categories)
    if selected_option and selected_option in words_array:
        df_neutral = st.session_state.output[st.session_state.output["emotion"] == selected_option]
        all_text = " ".join(df_neutral["text"].tolist())

        words = nltk.word_tokenize(all_text)

        stop_words = set(stopwords.words("english"))

        filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
        word_freq = Counter(filtered_words)
        top_20_words = word_freq.most_common(20)

        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(
            dict(top_20_words)
        )

        st.image(wordcloud.to_array())
        temp_df = pd.DataFrame(top_20_words, columns=["word", "frequency"])

        fig = px.bar(
            temp_df, x="word", y="frequency", text="frequency", labels={"word": "Word", "frequency": "Frequency"}
        )
        fig.update_traces(hovertemplate="Word: %{x}<br>Frequency: %{y}", hoverinfo="text")

        st.plotly_chart(fig, use_container_width=True)

        freq_words_array = [word for word, _ in top_20_words]
        freq_selected_word = st.selectbox("Select a word:", freq_words_array)
        if freq_selected_word:
            filtered_df = df_neutral[df_neutral["text"].str.contains(freq_selected_word)]
            filtered_df = filtered_df[["text", "emotion"]]
            st.dataframe(filtered_df)
