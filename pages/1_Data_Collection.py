import pkgutil

import pandas as pd
import streamlit as st

import data_collection
from data_collection.utils import download_images


def none_default_sidebar_text_input(label):
    v = st.sidebar.text_input(label)
    if v == "":
        return None
    return v


def query_builder(option):
    if option == "keywords":
        return none_default_sidebar_text_input("Keywords (Comma separated)")
    if option == "count":
        return st.sidebar.number_input("Number of posts", value=100)
    if option == "images":
        return st.sidebar.checkbox("Must have images")
    if option == "start_date":
        return st.sidebar.date_input("Start date")
    if option == "end_date":
        return st.sidebar.date_input("End date")
    if option == "locations":
        return none_default_sidebar_text_input("Locations e.g. US,MX")
    if option == "hashtags":
        return none_default_sidebar_text_input("Hashtags e.g. fun,comedy")
    if option == "video_url":
        return none_default_sidebar_text_input("Video URL")

    raise ValueError(f"Unknown query option: {option}")


title = "Data Collection"
# st.set_page_config(page_title=title)

st.sidebar.header(title)

if "results" not in st.session_state:
    st.session_state.results = None

# Grabs all the packages in the data_collection folder
social_medias_package_names = [
    name for _, name, is_pkg in pkgutil.walk_packages([data_collection.__path__[0]]) if is_pkg
]

social_medias = {}
for social_media_package_name in social_medias_package_names:
    social_media = getattr(data_collection, social_media_package_name)
    social_medias[social_media.name] = social_media

social_media = social_medias[st.sidebar.selectbox("Select a social media platform", social_medias.keys())]

collector = social_media.collection_methods[
    st.sidebar.selectbox("Select a collection type", social_media.collection_methods.keys())
].Collector()

st.sidebar.markdown("---")
st.sidebar.header("Query Options")

query_options = collector.query_options()
query_values = {}
for query_option in query_options:
    query_values[query_option] = query_builder(query_option)

if st.sidebar.button("Collect"):
    st.session_state.results = None
    with st.spinner("Collecting data..."):
        st.session_state.results = collector.collect(**query_values)

if st.session_state.results is not None:
    st.write("Found ", len(st.session_state.results), " results")
    df = pd.DataFrame(st.session_state.results)
    tabs = st.tabs(["Results", "Raw Data"])
    with tabs[0]:
        # Convert results to a dataframe
        st.dataframe(df)

    with tabs[1]:
        st.write(st.session_state.results)

    # Option to save
    if "keywords" in query_values.keys():
        save_name = st.text_input("Save name", value=f"{social_media.name}-{query_values['keywords']}")
    else:
        save_name = st.text_input("Save name", value=f"{social_media.name}-")
    do_download_images = st.checkbox("Download images with save")
    if do_download_images and "image_urls" not in df.columns:
        st.error("Cannot download images because the results do not have an 'image_urls' column")
    if st.button("Save"):
        df.to_csv(f"data/{save_name}.csv")
        st.success(f"Saved as data/{save_name}.csv")

        if do_download_images:
            download_images_progress_bar = st.progress(0)
            image_path = ""
            for i in range(len(st.session_state.results)):
                image_path = download_images(st.session_state.results, save_name, i)
                download_images_progress_bar.progress(
                    i / len(st.session_state.results),
                    text=f"Downloading images (images from {i + 1}/{len(st.session_state.results)} results downloaded)",
                )
            st.success("Successfully downloaded all the images to '" + image_path + "'")
