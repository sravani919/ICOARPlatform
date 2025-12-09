from collections import Counter
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud


# simple tokenizer without NLTK complications
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "this", "that", "it",
    "with", "as", "at", "by", "from", "about", "into", "up", "down",
    "out", "over", "under", "so", "if", "then", "than", "too", "very",
    "can", "could", "should", "would", "will", "just", "not", "no",
    "yes", "but"
}


def _tokenize(text: str):
    text = str(text).lower()
    tokens = re.findall(r"[a-zA-Z']+", text)
    return [t for t in tokens if t and t not in STOPWORDS]


def emotional_analysis(df: pd.DataFrame):
    # df is expected to already have 'text' and 'emotion' columns
    st.write("Incoming df columns:", list(df.columns))

    if not {"text", "emotion"}.issubset(df.columns):
        st.error(
            "emotional_analysis expects a dataframe with 'text' and 'emotion' columns.\n\n"
            f"Current columns: {list(df.columns)}"
        )
        return

    # optional: keep df in session if something else needs it
    st.session_state.output = df

    # 2) Determine which emotions are present in the data
    categories_all = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise",
        "neutral",
    ]

    df_emotions = df["emotion"].astype(str)
    present = set(df_emotions.unique().tolist())
    categories = [c for c in categories_all if c in present]
    if not categories:
        # fall back to whatever is there
        categories = sorted(present)

    if not categories:
        st.warning("No recognizable emotion values found in the dataset.")
        return

    # 3) Frequency distribution of emotions
    freq = [int((df_emotions == emotion).sum()) for emotion in categories]

    fig1 = go.Figure(
        data=[go.Pie(labels=categories, values=freq, textinfo="none")]
    )
    fig1.update_layout(title="Emotion Frequency Distribution", height=500, width=700)
    st.plotly_chart(fig1, use_container_width=True)

    # 4) Detailed view for a selected emotion
    st.subheader("Explore a specific emotion")
    selected_option = st.selectbox("Select an emotion:", categories)

    if not selected_option:
        return

    df_sel = df[df_emotions == selected_option]
    if df_sel.empty:
        st.info("No rows found for this emotion.")
        return

    # --- Wordcloud + Top words ---
    all_text = " ".join(df_sel["text"].astype(str).tolist())
    tokens = _tokenize(all_text)

    if not tokens:
        st.info("Not enough text to build a wordcloud / word frequency view.")
        return

    word_freq = Counter(tokens)
    top_20_words = word_freq.most_common(20)

    # Wordcloud
    st.subheader("Wordcloud")
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(dict(top_20_words))
    st.image(wc.to_array(), use_column_width=True)

    # Bar chart of word frequency
    temp_df = pd.DataFrame(top_20_words, columns=["word", "frequency"])

    fig = px.bar(
        temp_df,
        x="word",
        y="frequency",
        text="frequency",
        labels={"word": "Word", "frequency": "Frequency"},
    )
    fig.update_traces(
        hovertemplate="Word: %{x}<br>Frequency: %{y}",
        hoverinfo="text",
        textposition="outside",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5) Example posts for frequent words
    st.subheader("Example posts for frequent words")
    freq_words_array = [w for w, _ in top_20_words]
    freq_selected_word = st.selectbox("Select a word:", freq_words_array)

    if freq_selected_word:
        filtered_df = df_sel[
            df_sel["text"].str.contains(freq_selected_word, case=False, na=False)
        ][["text", "emotion"]]

        st.dataframe(filtered_df, use_container_width=True)
