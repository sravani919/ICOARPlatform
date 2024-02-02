import pkgutil

import pandas as pd
import streamlit as st

from ICOAR_core import data_collection
from ICOAR_core.data_collection.utils import ProgressUpdate, download_images

if "results" not in st.session_state:
    st.session_state.results = None
if "step" not in st.session_state:
    st.session_state.step = 0
if "collector" not in st.session_state:
    st.session_state.collector = None
if "socialmedia" not in st.session_state:
    st.session_state.socialmedia = None

# Option variables so when you go backwards in the stepper bar it remembers the options you selected
st.session_state.social_media_option = "Facebook"
st.session_state.collector_option = None
st.session_state.query_values = {}


def reset_query_values():
    st.session_state.query_values = {}


def none_default_text_input(label):
    """
    Creates a text input that returns None if the user inputs nothing
    :param label: The label of the text input
    :return: The value of the text input or None if the user inputs nothing
    """
    v = st.text_input(label)
    if v == "":
        return None
    return v


def query_builder(option, container=None):
    """
    Creates a query option based on the option and container and inserts it into the container.
    :param option: The query option to create
    :param container:  The container to insert the query option into
    :return:The value of the query option
    """
    with container:
        if option == "keywords" or option == "keywords AND":
            return none_default_text_input("Keywords AND (Comma separated)")
        if option == "keywordsOR":
            return none_default_text_input("Keywords OR (Comma separated)")
        if option == "count":
            return st.number_input("Number of posts", value=100)
        if option == "images":
            return st.checkbox("Must have images")
        if option == "start_date":
            return st.date_input("Start date")
        if option == "end_date":
            return st.date_input("End date")
        if option == "locations":
            return none_default_text_input("Locations e.g. US,MX")
        if option == "hashtags":
            return none_default_text_input("Hashtags e.g. fun,comedy")
        if option == "video_url":
            return none_default_text_input("Video URL")
        if option == "search_id":
            return none_default_text_input("Search ID")
        if option == "cursor":
            return none_default_text_input("Cursor")
        if option == "kaggle_dataset":
            # Prompt to visit kaggle.com/datasets and get the dataset owner and dataset name
            st.markdown("Visit [kaggle.com/datasets](https://www.kaggle.com/datasets)")
            st.markdown("---------------")
            return none_default_text_input("Kaggle dataset url")
        if option == "delete_temp_data":
            # Check box if the user wants to delete unsaved preview data
            return st.checkbox("Delete unsaved preview data afterwards")

    raise ValueError(f"Unknown query option: {option}")


def social_media_selector(social_medias):
    st.session_state.social_media_option = st.radio(
        "Platform", list(social_medias.keys()), on_change=reset_query_values
    )
    # index=list(social_medias.keys()).index(
    #     st.session_state.social_media_option))
    st.session_state.socialmedia = social_medias[st.session_state.social_media_option]


def collection_type_selector():
    if "collector_option" not in st.session_state:
        st.session_state.collector_option = None
    try:
        list(st.session_state.socialmedia.collection_methods.keys()).index(st.session_state.collector_option)
    except ValueError:
        pass
    st.session_state.collector_option = st.radio(
        "Collection type", list(st.session_state.socialmedia.collection_methods.keys()), on_change=reset_query_values
    )
    st.session_state.collector = st.session_state.socialmedia.collection_methods[
        st.session_state.collector_option
    ].Collector()


def data_collection_tab():
    if "results" not in st.session_state:
        st.session_state.results = None
    if "query_values" not in st.session_state:
        st.session_state.query_values = {}
    # Grabs all the packages in the data_collection folder
    social_medias_package_names = [
        name for _, name, is_pkg in pkgutil.walk_packages([data_collection.__path__[0]]) if is_pkg
    ]

    # Organizes the social media platforms into a dictionary
    # The dictionary is of the form {platform_name: platform_collector_object}
    social_medias = {}
    for social_media_package_name in social_medias_package_names:
        sm = getattr(data_collection, social_media_package_name)
        social_medias[sm.name] = sm

    # Stepper bar
    # steps = ["Select a social media platform", "Select a collection type", "Query options"]
    # st.session_state.step = stx.stepper_bar(steps=steps)
    main_columns = st.columns(3)

    with main_columns[0]:
        st.subheader("1. Select a platform")
        social_media_selector(social_medias)
        # Fancy vertical divider
        st.markdown("-------------------")
        collection_type_selector()

    with main_columns[1]:
        st.subheader("2. Query options")
        query_options = st.session_state.collector.query_options()
        columns = st.columns(2)
        for query_option in query_options:
            st.session_state.query_values[query_option] = query_builder(query_option, columns[0])

    with main_columns[2]:
        st.subheader("3. Summary")

        platform = st.session_state.socialmedia.name
        collection_method = st.session_state.collector.__class__.__module__.split(".")[-1].title()

        # Creating a nice Markdown table to display the summary of the current query
        summary = f"""
| Query option | Value |
| ------------ | ----- |
| :blue[Platform] | :blue[{platform}] |
| :blue[Collection method] | :blue[{collection_method}] |
"""
        for query_option in query_options:
            if query_option in st.session_state.query_values.keys():
                summary += f"| {query_option} |"
                if st.session_state.query_values[query_option] is None:
                    summary += ":red[None] |"
                else:
                    summary += f" {st.session_state.query_values[query_option]} |"
                summary += "\n"
        st.markdown(summary)
        st.markdown("-------------------")

        if st.button("Collect"):
            # The last yield contains the data
            # Every other yield contains a tuple with a float between 0 and 1 representing the progress and a string
            # describing the progress
            # Updating a progress bar with each yield until the last one
            # The last yield should be the data
            st.session_state.results = None
            gen = st.session_state.collector.collect_generator(**st.session_state.query_values)
            progress_bar = st.progress(0)
            for i, data in enumerate(gen):
                if isinstance(data, ProgressUpdate):
                    progress_bar.progress(data.progress, text=data.text)
                else:
                    st.session_state.results = data
                    break

    if st.session_state.results is not None:
        cols = st.columns(1)
        with cols[0]:
            st.subheader("Results")
            st.write("Found ", len(st.session_state.results), " results")
            df = pd.DataFrame(st.session_state.results)
            tabs = st.tabs(["Results", "Raw Data"])
            with tabs[0]:
                # Convert results to a dataframe
                st.dataframe(df)

            with tabs[1]:
                st.write(st.session_state.results)

            # If keywords is one of the query options, use it as the default save name
            if "keywords" in st.session_state.query_values.keys():
                save_name = st.text_input(
                    "Save name",
                    value=f"{st.session_state.socialmedia.name}-{st.session_state.query_values['keywords']}",
                )
            else:
                save_name = st.text_input("Save name", value=f"{st.session_state.socialmedia.name}-")
            do_download_images = st.checkbox("Download images with save")
            if do_download_images and "image_urls" not in df.columns:
                st.error("Cannot download images because the results do not have an 'image_urls' column")
            if st.button("Save"):
                data_collection.utils.save_data(st.session_state.results, save_name)
                username = st.session_state["username"]
                st.success(f"Saved as data/{username}/{save_name}.csv")

                if do_download_images:
                    download_images_progress_bar = st.progress(0)
                    image_path = ""
                    for i in range(len(st.session_state.results)):
                        image_path = download_images(st.session_state.results, save_name, i)
                        download_images_progress_bar.progress(
                            i / len(st.session_state.results),
                            text=f"Downloading images (images from {i + 1}/{len(st.session_state.results)} \
                            results downloaded)",
                        )
                    st.success("Successfully downloaded all the images to '" + image_path + "'")
