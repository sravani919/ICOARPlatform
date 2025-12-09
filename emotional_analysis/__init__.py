from collections import Counter
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

# Simple stopword list (no NLTK needed)
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "this", "that", "it",
    "with", "as", "at", "by", "from", "about", "into", "up", "down",
    "out", "over", "under", "so", "if", "then", "than", "too", "very",
    "can", "could", "should", "would", "will", "just", "not", "no",
    "yes", "but"
}


def _tokenize(text: str):
    """Simple tokenizer using regex + STOPWORDS."""
    text = str(text).lower()
    tokens = re.findall(r"[a-zA-Z']+", text)
    return [t for t in tokens if t and t not in STOPWORDS]


def emotional_analysis(df: pd.DataFrame):
    """
    df is expected to ALREADY have columns: 'text' and 'emotion'.

    Text_Visualisation_tab() is responsible for letting the user pick which
    original columns map to these, and renaming them before passing here.
    """
    st.write("Incoming df columns:", list(df.columns))

    # Sanity check
    if not {"text", "emotion"}.issubset(df.columns):
        st.error(
            "Emotion analysis expects a dataframe with 'text' and 'emotion' columns.\n\n"
            f"Current columns: {list(df.columns)}"
        )
        return

    # Drop rows with missing text/emotion
    df = df.dropna(subset=["text", "emotion"])
    if df.empty:
        st.error("No data available after removing rows with missing text/emotion.")
        return

    # optional: keep df in session
    st.session_state.output = df

    # ---------- 1) Emotion frequency using whatever labels exist ----------
    df_emotions = df["emotion"].astype(str)
    label_counts = df_emotions.value_counts()

    categories = label_counts.index.tolist()
    freq = label_counts.values.tolist()

    if not categories:
        st.warning("No emotion labels found in the dataset.")
        return

    fig1 = go.Figure(
        data=[go.Pie(labels=categories, values=freq, textinfo="none")]
    )
    fig1.update_layout(title="Emotion Frequency Distribution", height=500, width=700)
    st.plotly_chart(fig1, use_container_width=True)

    # ---------- 2) Detailed view for selected emotion ----------
    st.subheader("Explore a specific emotion")
    selected_option = st.selectbox("Select an emotion:", categories)

    if not selected_option:
        return

    df_sel = df[df_emotions == selected_option]
    if df_sel.empty:
        st.info("No rows found for this emotion.")
        return

    # ---------- 3) Wordcloud + Top words ----------
    all_text = " ".join(df_sel["text"].astype(str).tolist())
    tokens = _tokenize(all_text)

    if not tokens:
        st.info("Not enough text to build a wordcloud / word frequency view.")
        return

    word_freq = Counter(tokens)
    top_20_words = word_freq.most_common(20)

    st.subheader("Wordcloud")
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(dict(top_20_words))
    st.image(wc.to_array(), use_container_width=True)

    # ---------- 4) Bar chart of word frequency ----------
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

    # ---------- 5) Example posts for frequent words ----------
    st.subheader("Example posts for frequent words")
    freq_words_array = [w for w, _ in top_20_words]
    freq_selected_word = st.selectbox("Select a word:", freq_words_array)

    if freq_selected_word:
        filtered_df = df_sel[
            df_sel["text"].str.contains(freq_selected_word, case=False, na=False)
        ][["text", "emotion"]]

        st.dataframe(filtered_df, use_container_width=True)
