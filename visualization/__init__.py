from collections import Counter
from typing import Union, Optional
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

# simple built-in stopword list to avoid NLTK corpus issues
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "this", "that", "it",
    "with", "as", "at", "by", "from", "about", "into", "up", "down",
    "out", "over", "under", "so", "if", "then", "than", "too", "very",
    "can", "could", "should", "would", "will", "just", "not", "no",
    "yes", "but"
}


def _infer_label_col(value_counts: Union[pd.Series, dict],
                     data: pd.DataFrame) -> Optional[str]:
    """
    Try to infer which column in `data` produced `value_counts`.
    Assumes you did: value_counts = data[label_col].value_counts()
    so value_counts.name == label_col.
    """
    if isinstance(value_counts, pd.Series) and value_counts.name in data.columns:
        return value_counts.name

    # fallback: try some common label column names
    candidates = [
        "sentiment", "Sentiment",
        "label", "Label",
        "prediction", "pred",
        "sentiment_label", "emotion", "Emotion",
    ]
    for c in candidates:
        if c in data.columns:
            return c

    return None


def _tokenize(text: str):
    """Simple tokenizer using regex and our STOPWORDS (no NLTK)."""
    text = str(text).lower()
    tokens = re.findall(r"[a-zA-Z']+", text)
    return [t for t in tokens if t and t not in STOPWORDS]


def add_graph_info(value_counts: Union[pd.Series, dict],
                   data: pd.DataFrame) -> None:
    """
    Generic helper to show more information about the plotted labels.

    It works with ANY label column (sentiment, label, prediction, emotion, etc.)
    and assumes there is a 'text' column for examples / wordclouds.
    """
    # Ensure value_counts is a Series
    if not isinstance(value_counts, pd.Series):
        value_counts = pd.Series(value_counts)

    label_col = _infer_label_col(value_counts, data)

    # ---- high-level summary ----
    st.markdown("### Dataset summary")
    st.write(f"Total rows in dataset: **{len(data)}**")
    st.write(f"Number of unique labels in this plot: **{len(value_counts)}**")

    st.markdown("#### Label distribution")
    st.dataframe(
        value_counts.rename("count").to_frame(),
        use_container_width=True,
    )

    # If we can't map back to a specific column, stop here
    if label_col is None:
        st.info(
            "Detailed analysis is not available because the label column "
            "could not be inferred. Please make sure your label column "
            "has a clear name like 'sentiment', 'label', or 'prediction'."
        )
        return

    if "text" not in data.columns:
        st.info(
            "No 'text' column found in the dataset. "
            "Wordcloud and example texts are disabled."
        )
        return

    # ---- label-level exploration ----
    st.markdown("#### Explore a specific label")

    label_values = value_counts.index.tolist()
    selected_label = st.selectbox(
        "Select a label to inspect",
        options=[str(v) for v in label_values],
    )

    # Filter rows for the selected label
    mask = data[label_col].astype(str) == selected_label
    subset = data[mask]

    st.write(f"Number of rows for this label: **{len(subset)}**")

    if subset.empty:
        st.info("No rows found for this label.")
        return

    # ---- Wordcloud + Top words ----
    all_text = " ".join(subset["text"].astype(str).tolist())
    tokens = _tokenize(all_text)

    if not tokens:
        st.info("Not enough text to build a wordcloud / word frequency view.")
        return

    word_freq = Counter(tokens)
    top_20_words = word_freq.most_common(20)

    cols = st.columns(3)

    # Wordcloud
    with cols[0]:
        st.subheader("Wordcloud")
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate_from_frequencies(dict(top_20_words))
        st.image(wc.to_array(), use_column_width=True)

    # Bar chart of word frequencies
    with cols[1]:
        st.subheader("Words Frequency")
        temp_df = pd.DataFrame(top_20_words, columns=["word", "frequency"])
        fig = px.bar(
            temp_df,
            x="word",
            y="frequency",
            text="frequency",
            labels={"word": "Word", "frequency": "Frequency"},
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate="Word: %{x}<br>Frequency: %{y}",
            hoverinfo="text",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Examples containing a chosen word
    with cols[2]:
        st.subheader("Most used words – example texts")
        freq_words_array = [w for w, _ in top_20_words]
        freq_selected_word = st.selectbox(
            "Select a word to show examples:",
            freq_words_array,
        )

        if freq_selected_word:
            filtered_df = subset[
                subset["text"].str.contains(freq_selected_word, case=False, na=False)
            ][["text", label_col]]
            st.dataframe(filtered_df, use_container_width=True)
