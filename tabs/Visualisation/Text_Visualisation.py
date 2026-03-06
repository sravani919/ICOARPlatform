import glob

import gensim
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import streamlit as st

from emotional_analysis import emotional_analysis
from tabs.Data_Collection.data_upload import data_upload_element
from visualization import add_graph_info


# -----------------------------
# UI helpers (same style as DC / PP)
# -----------------------------
def _viz_hero_header():
    steps = [
        (1, "Select data"),
        (2, "Choose chart"),
        (3, "Generate"),
        (4, "Export"),
    ]
    active = int(st.session_state.get("viz_step", 1))

    chips_html = '<div class="viz-steps">' + "".join(
        [
            f'<div class="viz-chip {"active" if s==active else ""}">'
            f'<span class="viz-chip-num">{s}</span>'
            f'<span class="viz-chip-name">{name}</span>'
            f"</div>"
            for s, name in steps
        ]
    ) + "</div>"

    st.markdown(
        f"""
<div class="viz-hero-bleed">
  <div class="viz-hero-inner">
    <div class="viz-hero-title">Visualization</div>
    <div class="viz-hero-sub">Pick a dataset, choose a visualization type, and explore insights.</div>
    {chips_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _viz_step_card(title: str):
    wrap = st.container()
    with wrap:
        st.markdown('<div class="viz-card-marker"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="viz-step-title">{title}</div>', unsafe_allow_html=True)
    return wrap


def _pick_label_column(df_):
    preferred = [
        "sentiment", "Sentiment",
        "label", "Label",
        "prediction", "pred",
        "sentiment_label",
    ]
    cols = list(df_.columns)
    for c in preferred:
        if c in cols:
            return cols.index(c)
    return 0


# -----------------------------
# Main Tab
# -----------------------------
def Text_Visualisation_tab():
    # defaults
    if "filename_pred" not in st.session_state:
        st.session_state.filename_pred = ""
    if "viz_step" not in st.session_state:
        st.session_state.viz_step = 1

    username = st.session_state.get("username", "")

    _viz_hero_header()
    st.markdown('<div class="viz-wrap">', unsafe_allow_html=True)

    # =========================================================
    # STEP 1 — SELECT DATA
    # =========================================================
    if st.session_state.viz_step == 1:
        card = _viz_step_card("1) Select data")

        with card:
            # Prefer cleaned df from preprocessing if available
            processed_df = st.session_state.get("processed_df", None)
            if processed_df is not None:
                st.success("Using cleaned dataset from Pre-processing.")
                st.dataframe(processed_df, use_container_width=True, height=320)

                spacer, next_col = st.columns([0.76, 0.24])
                with next_col:
                    if st.button("Next →", type="primary", use_container_width=True):
                        st.session_state.viz_step = 2
                        st.rerun()

            else:
                source = st.radio(
                    "Choose data source:",
                    ("Upload / Select saved dataset", "Select from predicted folder"),
                    horizontal=True,
                )

                if source == "Upload / Select saved dataset":
                    st.session_state.filename_pred = ""
                    uploaded_file = data_upload_element(username, get_filepath_instead=True)
                    if uploaded_file:
                        st.session_state.filename_pred = uploaded_file

                else:
                    folder_files = [file for file in glob.glob(f"./predicted/{username}/*.csv")]
                    selected_file = st.selectbox("Select a file from folder", [""] + folder_files)

                    if st.button("Load", use_container_width=False):
                        if selected_file:
                            st.session_state.filename_pred = selected_file

                if not st.session_state.filename_pred:
                    st.info("Select a dataset to continue.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                # load and preview
                try:
                    df = pd.read_csv(st.session_state.filename_pred)
                    st.session_state["viz_df"] = df
                    st.dataframe(df, use_container_width=True, height=320)
                except Exception as e:
                    st.error("Failed to read the selected CSV.")
                    st.exception(e)
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                spacer, next_col = st.columns([0.76, 0.24])
                with next_col:
                    if st.button("Next →", type="primary", use_container_width=True):
                        st.session_state.viz_step = 2
                        st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        return  # stop here in step 1

    # =========================================================
    # STEP 2+ — VISUALIZATIONS (your existing logic)
    # =========================================================
    card = _viz_step_card("2) Choose a type of visualisation")

    with card:
        # get df either from preprocessing or from selected file
        data = st.session_state.get("processed_df", None)
        if data is None:
            data = st.session_state.get("viz_df", None)

        if data is None:
            st.warning("No dataset loaded. Go back and select data.")
            if st.button("← Back", use_container_width=False):
                st.session_state.viz_step = 1
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            return

        options = [
            "📊Bar Plot",
            "🥧Pie Chart",
            "🎯Topic Modeling",
            "📈Temporal Analysis",
            "Emotion Analysis",
        ]
        selected_option = st.selectbox("Select a type of visualisation", options)

        # For some visualizations, we DO need a text-like column
        text_required_options = ["🎯Topic Modeling", "📈Temporal Analysis", "Emotion Analysis"]
        if selected_option in text_required_options and "text" not in data.columns:
            st.error(
                "The selected dataset must contain a 'text' column for this visualisation "
                "(Topic Modeling, Temporal Analysis, or Emotion Analysis)."
            )
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if "text" in data.columns:
            data = data[data["text"].notna()]  # remove rows with NaN text

        # ====================== 📊 BAR PLOT ======================
        if selected_option == "📊Bar Plot":
            cols_all = list(data.columns)
            label_index = _pick_label_column(data)
            label_col = st.selectbox(
                "Select label column for bar plot",
                cols_all,
                index=min(label_index, len(cols_all) - 1),
            )

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
            st.plotly_chart(fig1, use_container_width=True)

            with st.expander("Show Additional Information"):
                add_graph_info(value_counts, data)

        # ====================== 🥧 PIE CHART ======================
        elif selected_option == "🥧Pie Chart":
            cols_all = list(data.columns)
            label_index = _pick_label_column(data)
            label_col = st.selectbox(
                "Select label column for pie chart",
                cols_all,
                index=min(label_index, len(cols_all) - 1),
            )

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
            st.plotly_chart(fig1, use_container_width=True)

            with st.expander("Show Additional Information"):
                add_graph_info(value_counts, data)

        # ====================== 🎯 TOPIC MODELING ======================
        elif selected_option == "🎯Topic Modeling":
            data_text = data["text"].astype(str).tolist()
            tokenized_data = [txt.split() for txt in data_text]

            if not tokenized_data:
                st.warning("No text data available for topic modeling.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            dictionary = gensim.corpora.Dictionary(tokenized_data)
            corpus = [dictionary.doc2bow(text) for text in tokenized_data]

            if len(dictionary) == 0:
                st.warning("Not enough vocabulary in the text to build topics.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            lda_model = gensim.models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=10,
                passes=10,
            )

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

            sample_key = {"1 day": "D", "1 hour": "60min", "30 minutes": "30min", "1 minute": "1min"}.get(
                time_interval, "D"
            )

            # date column
            if "date" in data.columns:
                date_key = "date"
            elif "create_time" in data.columns:
                date_key = "create_time"
            else:
                st.error("The date or create_time column is not present in the dataset.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            # user column
            if "user_name" in data.columns:
                user_key = "user_name"
            elif "username" in data.columns:
                user_key = "username"
            elif "author_username" in data.columns:
                user_key = "author_username"
            else:
                st.error("The user_name, username or author_username column is not present in the dataset.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            data.loc[:, date_key] = pd.to_datetime(data[date_key], errors="coerce")
            data = data[data[date_key].notna()]

            keywords = (
                st.text_input("Enter the keywords of interest separated by comma (i.e., covid, lockdown, ... ):")
                .lower()
                .split(",")
            )
            keywords = [k.strip() for k in keywords if k.strip()]
            if not keywords:
                st.info("Enter at least one keyword to plot temporal trends.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            masks = [data["text"].str.contains(keyword, case=False, na=False) for keyword in keywords]

            daily_counts = [
                data.loc[mask, date_key].value_counts().sort_index().resample(sample_key).sum()
                for mask in masks
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
                    fig_counts.add_trace(go.Scatter(x=count.index, y=count.values, mode="lines", name=keywords[i].capitalize()))
                st.plotly_chart(fig_counts, use_container_width=True)

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
                    fig_top_posters.add_trace(go.Bar(x=top_posters.index, y=top_posters.values, name=keywords[i].capitalize()))
                st.plotly_chart(fig_top_posters, use_container_width=True)

            cols2 = st.columns(2)
            with cols2[0]:
                mask = data["text"].str.contains("|".join(keywords), case=False, na=False)
                daily_counts_sum = data.loc[mask, date_key].value_counts().sort_index().resample(sample_key).sum()
                fig1 = go.Figure(
                    data=[go.Scatter(x=daily_counts_sum.index, y=daily_counts_sum.values, mode="lines", name="Daily Count")],
                    layout=go.Layout(title="Daily Tweet Counts", xaxis=dict(title="Date"), yaxis=dict(title="Count")),
                )
                st.plotly_chart(fig1, use_container_width=True)

            with cols2[1]:
                top_posters = data[user_key].value_counts().nlargest(10)
                fig2 = go.Figure(
                    data=[go.Bar(x=top_posters.index, y=top_posters.values, name="Top 10 Posters")],
                    layout=go.Layout(title="Top 10 Posters", xaxis=dict(title="User"), yaxis=dict(title="Tweet Count")),
                )
                st.plotly_chart(fig2, use_container_width=True)

        # ====================== 😊 EMOTION ANALYSIS ======================
        elif selected_option == "Emotion Analysis":
            st.subheader("Emotion Analysis")

            cols_all = list(data.columns)

            text_col = st.selectbox(
                "Select the text column",
                cols_all,
                index=cols_all.index("text") if "text" in cols_all else 0,
            )

            emotion_col = st.selectbox(
                "Select the emotion/label column",
                cols_all,
                index=cols_all.index("emotion") if "emotion" in cols_all else 0,
            )

            df_emotion = data[[text_col, emotion_col]].rename(columns={text_col: "text", emotion_col: "emotion"})
            emotional_analysis(df_emotion)

        # Navigation buttons at bottom
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        spacer, back_col = st.columns([0.82, 0.18])
        with back_col:
            if st.button("← Back", use_container_width=True):
                st.session_state.viz_step = 1
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)