# Path: /home/spati/ICOAR/visualization/visualizer.py

from collections import Counter
import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.corpus import stopwords
from wordcloud import WordCloud

def add_graph_info(value_counts, data):
    selected_category = st.selectbox("Select a category", value_counts.index)
    words_array = data["sentiment"].unique()

    # see more information about the selected category, wordcloud, most used words etc.
    if selected_category and selected_category in words_array:
        cols = st.columns(3)
        df_neutral = data[data["sentiment"] == selected_category]
        all_text = " ".join(df_neutral["text"].tolist())

        words = nltk.word_tokenize(all_text)

        stop_words = set(stopwords.words("english"))

        filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
        word_freq = Counter(filtered_words)
        top_20_words = word_freq.most_common(20)

        with cols[0]:
            st.subheader("Wordcloud")
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(
                dict(top_20_words)
            )

            st.image(wordcloud.to_array(), use_column_width=True)

        with cols[1]:
            st.subheader("Words Frequency")
            temp_df = pd.DataFrame(top_20_words, columns=["word", "frequency"])

            fig = px.bar(
                temp_df, x="word", y="frequency", text="frequency", labels={"word": "Word", "frequency": "Frequency"}
            )
            fig.update_traces(hovertemplate="Word: %{x}<br>Frequency: %{y}", hoverinfo="text")

            st.plotly_chart(fig, use_container_width=True)

        with cols[2]:
            st.subheader("Most used words examples")
            # user chooses from the 20 most common words and the posts with that word are displayed
            freq_words_array = [word for word, _ in top_20_words]
            freq_selected_word = st.selectbox("Select a word to show examples:", freq_words_array)
            if freq_selected_word:
                filtered_df = df_neutral[df_neutral["text"].str.contains(freq_selected_word)]
                filtered_df = filtered_df[["text", "sentiment"]]
                st.dataframe(filtered_df)

