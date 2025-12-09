import glob

import gensim
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import streamlit as st
# from nltk.tokenize import word_tokenize  # no longer needed

from emotional_analysis import emotional_analysis
from tabs.Data_Collection.data_upload import data_upload_element
from visualization import add_graph_info



def Text_Visualisation_tab():
    if "filename_pred" not in st.session_state:
        st.session_state.filename_pred = ""

    username = st.session_state["username"]

    # Use data_upload_element to allow file upload
    choice = st.radio("Choose data source:", ("Upload new data", "Select from folder"))

    if choice == "Upload new data":
        # Use data_upload to allow file upload
        st.session_state.filename_pred = ""
        uploaded_file = data_upload_element(username, get_filepath_instead=True)

        if uploaded_file:
            st.session_state.filename_pred = uploaded_file

    elif choice == "Select from folder":
        # Allow user to select a file from a folder
        folder_files = [file for file in glob.glob(f"./predicted/{username}/*.csv")]
        selected_file = st.selectbox("Select a file from folder", [""] + folder_files)

        if st.button("Load"):
            if selected_file:
                st.session_state.filename_pred = selected_file

    if not st.session_state.filename_pred:
        return

    # ----------------- Load data -----------------
    df = pd.read_csv(st.session_state.filename_pred)
    options = ["📊Bar Plot", "🥧Pie Chart", "🎯Topic Modeling", "📈Temporal Analysis", "Emotion Analysis"]
    selected_option = st.selectbox("Select an type of visualisation", options)

    data = df

    # Make sure we have text
    if "text" not in data.columns:
        st.error("The selected dataset must contain a 'text' column.")
        return

    data = data[data["text"].notna()]  # remove all data with nan text

    # ---------- helper to pick a sensible default label column ----------
    def pick_label_column(df_):
        preferred = ["sentiment", "Sentiment", "label", "Label", "prediction", "pred", "sentiment_label"]
        cols = list(df_.columns)
        for c in preferred:
            if c in cols:
                return cols.index(c)
        return 0  # fallback: first column
    # -------------------------------------------------------------------

    # ====================== 📊 BAR PLOT ======================
    if selected_option == "📊Bar Plot":
        cols_all = list(data.columns)
        label_index = pick_label_column(data)
        label_col = st.selectbox("Select label column for bar plot", cols_all, index=label_index)

        value_counts = data[label_col].value_counts()

        with st.expander("Show more graph options"):
            cols = st.columns(3)
            with cols[0]:
                title = st.text_input("Title", f"Classification of Posts by {label_col}")

                x_label = st.text_input("X label", label_col)
                y_label = st.text_input("Y label", "Count")
                label_font_size = st.slider("Label font size", 10, 50, 15)
            with cols[1]:
                title_font_size = st.slider("Title font size", 10, 50, 20)

                x_tick_font_size = st.slider("X tick font size", 5, 50, 10)
                y_tick_font_size = st.slider("Y tick font size", 5, 50, 10)
            with cols[2]:
                bar_color = st.color_picker("Bar color", "#1f77b4")
                outer_background_color = st.color_picker("Outer background color", "#FFFFFF")
                inner_background_color = st.color_picker("Inner background color", "#FFFFFF")
                text_color = st.color_picker("Text Color", "#000000")

        fig1 = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values)])
        fig1.update_traces(marker_color=bar_color)

        fig1.update_layout(
            title=title,
            title_font=dict(size=title_font_size, color=text_color),
            xaxis_title=x_label,
            yaxis_title=y_label,
            title_font_size=title_font_size,
            xaxis=dict(
                tickfont=dict(size=x_tick_font_size, color=text_color),
                title_font=dict(size=label_font_size, color=text_color),
            ),
            yaxis=dict(
                tickfont=dict(size=y_tick_font_size, color=text_color),
                title_font=dict(size=label_font_size, color=text_color),
            ),
            plot_bgcolor=inner_background_color,
            paper_bgcolor=outer_background_color,
            height=500,
            width=700,
        )
        st.plotly_chart(fig1)

        with st.expander("Show Additional Information"):
            add_graph_info(value_counts, data)

    # ====================== 🥧 PIE CHART ======================
    elif selected_option == "🥧Pie Chart":
        cols_all = list(data.columns)
        label_index = pick_label_column(data)
        label_col = st.selectbox("Select label column for pie chart", cols_all, index=label_index)

        value_counts = data[label_col].value_counts()

        with st.expander("Show more graph options"):
            cols = st.columns(2)
            with cols[0]:
                title = st.text_input("Title", f"Pie Chart of {label_col}")
                title_font_size = st.slider("Title font size", 10, 50, 20)
                text_color = st.color_picker("Text Color", "#000000")
            with cols[1]:
                label_font_size = st.slider("Label font size", 10, 50, 15)
                legend_font_size = st.slider("Legend font size", 10, 50, 15)
                background_color = st.color_picker("Background color", "#FFFFFF")

        fig1 = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values)])

        fig1.update_layout(
            title=title,
            title_font=dict(size=title_font_size, color=text_color),
            paper_bgcolor=background_color,
            font=dict(color=text_color),
            legend=dict(font=dict(size=legend_font_size, color=text_color)),
        )
        fig1.update_traces(
            textfont_size=label_font_size,
            hoverinfo="label+percent",
            texttemplate="%{label}<br>%{percent}",
            textposition="outside",
        )
        st.plotly_chart(fig1)
        with st.expander("Show Additional Information"):
            add_graph_info(value_counts, data)

    # ====================== 🎯 TOPIC MODELING ======================
    elif selected_option == "🎯Topic Modeling":
        # Avoid NLTK 'punkt_tab' issue: use simple whitespace split
        data_text = data["text"].astype(str).tolist()
        tokenized_data = [txt.split() for txt in data_text]

        if not tokenized_data:
            st.warning("No text data available for topic modeling.")
            return

        dictionary = gensim.corpora.Dictionary(tokenized_data)
        corpus = [dictionary.doc2bow(text) for text in tokenized_data]

        if len(dictionary) == 0:
            st.warning("Not enough vocabulary in the text to build topics.")
            return

        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

        topic_labels = []
        topics = lda_model.show_topics(num_topics=10, num_words=10, formatted=False)
        for topic in topics:
            words = [word for word, _ in topic[1]]
            topic_labels.append(" ".join(words))

        for i, label in enumerate(topic_labels):
            st.write(f"Topic {i + 1}: {label}")

        vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
        html_string = pyLDAvis.prepared_data_to_html(vis_data)

        st.components.v1.html(html_string, width=1480, height=960, scrolling=True)

    # ====================== 📈 TEMPORAL ANALYSIS ======================
    elif selected_option == "📈Temporal Analysis":
        interval_options = ["1 day", "1 hour", "30 minutes", "1 minute"]
        time_interval = st.selectbox("Select the time interval of the analysis:", interval_options)
        if time_interval == "1 day":
            sample_key = "D"
        elif time_interval == "1 hour":
            sample_key = "60min"
        elif time_interval == "30 minutes":
            sample_key = "30min"
        elif time_interval == "1 minute":
            sample_key = "1min"
        else:
            sample_key = "D"
        fig, ax = plt.subplots(figsize=(10, 6))
        # look for columns date or create_time, if not present, show error
        if "date" in data.columns:
            date_key = "date"
        elif "create_time" in data.columns:
            date_key = "create_time"
        else:
            st.error("The date or create_time column is not present in the dataset.")
            return
        if "user_name" in data.columns:
            user_key = "user_name"
        elif "username" in data.columns:
            user_key = "username"
        elif "author_username" in data.columns:
            user_key = "author_username"
        else:
            st.error("The user_name, username or author_username column is not present in the dataset.")
            return

        data.loc[:, date_key] = pd.to_datetime(data[date_key], errors="coerce")
        data = data[data[date_key].notna()]
        keywords = (
            st.text_input("Enter the keywords of interest seperated by comma (i.e., covid, lockdown, ... ):")
            .lower()
            .split(",")
        )

        masks = [data["text"].str.contains(keyword.strip(), case=False, na=False) for keyword in keywords]

        daily_counts = [
            data.loc[mask, date_key].value_counts().sort_index().resample(sample_key).sum() for mask in masks
        ]

        cols1 = st.columns(2)
        with cols1[0]:
            layout_counts = go.Layout(
                title="Daily Text Pattern Counts",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Count"),
                legend=dict(orientation="h"),
            )

            fig_counts = go.Figure(layout=layout_counts)

            for i, count in enumerate(daily_counts):
                fig_counts.add_trace(
                    go.Scatter(x=count.index, y=count.values, mode="lines", name=keywords[i].capitalize())
                )

            st.plotly_chart(fig_counts)

        with cols1[1]:
            layout_top_posters = go.Layout(
                title="Top Posters for Each Keyword",
                xaxis=dict(title="User"),
                yaxis=dict(title="Count"),
                barmode="group",
            )

            fig_top_posters = go.Figure(layout=layout_top_posters)

            for i, mask in enumerate(masks):
                top_posters = data.loc[mask, user_key].value_counts().nlargest(10)
                fig_top_posters.add_trace(
                    go.Bar(x=top_posters.index, y=top_posters.values, name=keywords[i].capitalize())
                )

            st.plotly_chart(fig_top_posters)

        cols2 = st.columns(2)
        with cols2[0]:
            # overall daily tweet count for all keywords
            mask = data["text"].str.contains("|".join(keywords), case=False)
            daily_counts_sum = data.loc[mask, date_key].value_counts().sort_index().resample(sample_key).sum()
            trace1 = go.Scatter(
                x=daily_counts_sum.index, y=daily_counts_sum.values, mode="lines", name="Daily Tweet Count"
            )

            layout1 = go.Layout(title="Daily Tweet Counts", xaxis=dict(title="Date"), yaxis=dict(title="Count"))

            fig1 = go.Figure(data=[trace1], layout=layout1)

            st.plotly_chart(fig1)

        with cols2[1]:
            top_posters = data[user_key].value_counts().nlargest(10)

            trace2 = go.Bar(x=top_posters.index, y=top_posters.values, name="Top 10 Posters")

            layout2 = go.Layout(title="Top 10 Posters", xaxis=dict(title="User"), yaxis=dict(title="Tweet Count"))

            fig2 = go.Figure(data=[trace2], layout=layout2)

            st.plotly_chart(fig2)

    # ====================== 😊 EMOTION ANALYSIS ======================
    elif selected_option == "Emotion Analysis":
        emotional_analysis(df)

