import glob

import gensim
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import seaborn as sns
import streamlit as st
from nltk.tokenize import word_tokenize

if "filename_pred" not in st.session_state:
    st.session_state.filename_pred = ""

option = st.selectbox("Select a file", [file for file in glob.glob("./predicted/*.csv")])

if st.button("Load"):
    st.session_state.filename_pred = option

if st.session_state.filename_pred != "":
    df = pd.read_csv(st.session_state.filename_pred)
    options = ["Topic Modeling", "Temporal Analysis", "User Network", "Bar Plot", "Pie Chart"]
    selected_option = st.selectbox("Select an type of visualisation", options)
    data = df
    fig, ax = plt.subplots()

    if selected_option == "Bar Plot":
        fig, ax = plt.subplots()
        value_counts = data["sentiment"].value_counts()
        bars = ax.bar(value_counts.index, value_counts.values)

        with st.expander("Show more graph options"):
            cols = st.columns(3)
            with cols[0]:
                title = st.text_input("Title", "Classification of tweets")
                title_font_size = st.slider("Title font size", 5, 50, 20)

            with cols[1]:
                x_label = st.text_input("X label", "Sentiment")
                y_label = st.text_input("Y label", "Count")
                label_font_size = st.slider("Label font size", 5, 50, 15)

            with cols[2]:
                x_tick_font_size = st.slider("X tick font size", 5, 50, 10)
                y_tick_font_size = st.slider("Y tick font size", 5, 50, 10)
                outer_background_color = st.color_picker("Outer background color", "#FFFFFF")
                inner_background_color = st.color_picker("Inner background color", "#FFFFFF")

        ax.tick_params(axis="x", labelsize=x_tick_font_size)
        ax.tick_params(axis="y", labelsize=y_tick_font_size)

        ax.set_xlabel(x_label, fontsize=label_font_size)
        ax.set_ylabel(y_label, fontsize=label_font_size)
        ax.set_title(title, fontsize=title_font_size)

        # Change the background color that's between the bars
        ax.set_facecolor(inner_background_color)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height}",
                ha="center",
                va="bottom",
                fontsize=12,  # Adjust the font size of the value labels
                color="black",  # Set the color of the value labels
            )

        st.pyplot(fig, facecolor=outer_background_color)

        with st.expander("Show Specific Posts"):
            selected_category = st.selectbox("Select a category", value_counts.index)
            tweets_in_category = data[data["sentiment"] == selected_category]

            st.write("Tweets in the selected category:")
            st.dataframe(tweets_in_category.reset_index(drop=True))
    elif selected_option == "Pie Chart":
        fig, ax = plt.subplots()
        value_counts = data["sentiment"].value_counts()
        ax.pie(value_counts.values, labels=value_counts.index, autopct="%1.1f%%")
        with st.expander("Show more graph options"):
            cols = st.columns(2)
            with cols[0]:
                title = st.text_input("Title", "Pie Chart")
                title_font_size = st.slider("Title font size", 10, 50, 20)
            with cols[1]:
                label_font_size = st.slider("Label font size", 10, 50, 15)
                percentages_font_size = st.slider("Percentages font size", 10, 50, 15)
                background_color = st.color_picker("Background color", "#FFFFFF")

        for text in ax.texts:
            if "%" in text.get_text():
                text.set_fontsize(percentages_font_size)
            else:
                text.set_fontsize(label_font_size)

        ax.set_title(title, fontsize=title_font_size)
        st.pyplot(fig, facecolor=background_color)

        with st.expander("Show Specific Posts"):
            selected_category = st.selectbox("Select a category", value_counts.index)
            tweets_in_category = data[data["sentiment"] == selected_category]

            st.write("Tweets in the selected category:")
            st.dataframe(tweets_in_category.reset_index(drop=True))
    elif selected_option == "Topic Modeling":
        data = data["text"].tolist()

        tokenized_data = [word_tokenize(text) for text in data]

        dictionary = gensim.corpora.Dictionary(tokenized_data)
        corpus = [dictionary.doc2bow(text) for text in tokenized_data]

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

        st.components.v1.html(html_string, width=1000, height=600, scrolling=True)

    elif selected_option == "Temporal Analysis":
        fig, ax = plt.subplots(figsize=(10, 6))
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

        data = data[data["date"].notna()]

        keywords = (
            st.text_input("Enter the keywords of interest seperated by comma (i.e., covid, lockdown, ... ):")
            .lower()
            .split(",")
        )

        masks = [data["text"].str.contains(keyword.strip(), case=False, na=False) for keyword in keywords]

        daily_counts = [data.loc[mask, "date"].value_counts().sort_index().resample("D").sum() for mask in masks]

        layout_counts = go.Layout(
            title="Daily Text Pattern Counts",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Count"),
            legend=dict(orientation="h"),
        )

        fig_counts = go.Figure(layout=layout_counts)

        for i, count in enumerate(daily_counts):
            fig_counts.add_trace(go.Scatter(x=count.index, y=count.values, mode="lines", name=keywords[i].capitalize()))

        st.plotly_chart(fig_counts)

        layout_top_posters = go.Layout(
            title="Top Posters for Each Keyword", xaxis=dict(title="User"), yaxis=dict(title="Count"), barmode="group"
        )

        fig_top_posters = go.Figure(layout=layout_top_posters)

        for i, mask in enumerate(masks):
            top_posters = data.loc[mask, "user_name"].value_counts().nlargest(10)
            fig_top_posters.add_trace(go.Bar(x=top_posters.index, y=top_posters.values, name=keywords[i].capitalize()))

        st.plotly_chart(fig_top_posters)

        trace1 = go.Scatter(x=daily_counts.index, y=daily_counts.values, mode="lines", name="Daily Tweet Count")

        layout1 = go.Layout(title="Daily Tweet Counts", xaxis=dict(title="Date"), yaxis=dict(title="Count"))

        fig1 = go.Figure(data=[trace1], layout=layout1)

        st.plotly_chart(fig1)

        top_posters = data["user_name"].value_counts().nlargest(10)

        trace2 = go.Bar(x=top_posters.index, y=top_posters.values, name="Top 10 Posters")

        layout2 = go.Layout(title="Top 10 Posters", xaxis=dict(title="User"), yaxis=dict(title="Tweet Count"))

        fig2 = go.Figure(data=[trace2], layout=layout2)

        st.plotly_chart(fig2)

    elif selected_option == "User Network":
        fig, ax = plt.subplots(figsize=(10, 6))
        keywords = st.text_input("Enter the keywords (comma-separated): ")
        keywords = [keyword.strip() for keyword in keywords.split(",")]

        G = nx.Graph()

        # Define color palette
        colors = sns.color_palette("Set2", len(keywords))

        filtered_data = data[data["text"].str.contains("|".join(keywords), case=False)]

        # Iterate over each filtered row
        for index, row in filtered_data.iterrows():
            text = row["text"]
            user = row["author_username"]

            # Check which keyword(s) are present in the text
            present_keywords = [keyword.capitalize() for keyword in keywords if keyword.lower() in text.lower()]

            # Add edges for the present keywords
            for keyword in present_keywords:
                G.add_edge(user, keyword, color=colors[keywords.index(keyword.lower())])

        # Position nodes using Fruchterman-Reingold layout
        pos = nx.fruchterman_reingold_layout(G)

        # Draw edges with colors
        edges = [(u, v) for (u, v, d) in G.edges(data=True)]

        # Draw the edges
        for edge in edges:
            color = G[edge[0]][edge[1]]["color"]
            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=color, alpha=0.5)

        # Draw the nodes (hidden in this case)
        nx.draw_networkx_nodes(G, pos, node_color="white", node_size=0)
        plt.axis("off")

        # Create legends
        legends = [
            plt.Line2D([], [], color=colors[i], alpha=0.5, label=keywords[i].capitalize()) for i in range(len(keywords))
        ]

        # Add legends to the plot
        plt.legend(handles=legends, loc="upper right")

        # Show the graph
        st.pyplot(fig)
