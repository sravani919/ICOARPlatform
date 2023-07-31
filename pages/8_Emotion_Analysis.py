import glob
from collections import Counter

import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords
from transformers import pipeline
from wordcloud import WordCloud

if "output" not in st.session_state:
    st.session_state.output = pd.DataFrame()
if "predict" not in st.session_state:
    st.session_state.predict = False
if "freq" not in st.session_state:
    st.session_state.freq = []

FILE = st.selectbox("Select a file", [file for file in glob.glob("./data/*.csv")])
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

freq = [0] * 28

if st.button("Predict"):
    df = pd.read_csv(FILE)
    emotion = pipeline("sentiment-analysis", model="arpanghoshal/EmoRoBERTa")
    df["emotion"] = ""

    for index, row in df.iterrows():
        pred = emotion(row["text"])[0]["label"]
        df.loc[index, "emotion"] = pred

        index = categories.index(pred)
        freq[index] += 1

    st.session_state.output = df
    st.session_state.predict = True
    st.session_state.freq = freq


if st.button("Reset"):
    st.session_state.predict = False
    st.session_state.freq = []
    st.session_state.output = pd.DataFrame()


if st.session_state.predict:
    fig1 = go.Figure(data=[go.Pie(labels=categories, values=st.session_state.freq, textinfo="none")])

    fig1.update_layout(title="Frequency Distribution", height=500, width=700)

    st.plotly_chart(fig1)

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
