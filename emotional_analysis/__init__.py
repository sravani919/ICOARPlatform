from collections import Counter

import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords
from wordcloud import WordCloud


def emotional_analysis(df: pd.DataFrame):
    import streamlit as st

    # 🔍 Show what columns we actually have
    st.write("Incoming df columns:", list(df.columns))

    # 1) Check required columns
    if "emotion" not in df.columns:
        st.error(
            "Emotion Analysis requires an 'emotion' column in the dataset.\n\n"
            f"Current columns: {list(df.columns)}"
        )
        return

    if "text" not in df.columns:
        st.error(
            "Emotion Analysis also needs a 'text' column to show examples and wordclouds.\n\n"
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
        # 🔍 Extra safety so we don't crash with KeyError
        st.write("Columns in df_sel:", list(df_sel.columns))

        required_cols = {"text", "emotion"}
        missing = required_cols - set(df_sel.columns)
        if missing:
            st.error(
                f"Expected columns {required_cols}, but missing: {missing}. "
                f"Available columns: {list(df_sel.columns)}"
            )
            return

        filtered_df = df_sel[
            df_sel["text"].str.contains(freq_selected_word, case=False, na=False)
        ][["text", "emotion"]]

        st.dataframe(filtered_df, use_container_width=True)
