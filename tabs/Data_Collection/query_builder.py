import streamlit as st


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

    def keywords_and():
        return none_default_text_input("Keywords AND (Comma separated)")

    def keywords_or():
        return none_default_text_input("Keywords OR (Comma separated)")

    def count():
        return st.number_input("Number of posts", value=100)

    def images():
        return st.checkbox("Must have images")

    def start_date():
        return st.date_input("Start date")

    def end_date():
        return st.date_input("End date")

    def locations():
        return none_default_text_input("Locations e.g. US,MX")

    def hashtags():
        return none_default_text_input("Hashtags e.g. fun,comedy")

    def video_url():
        return none_default_text_input("Video URL")

    def search_id():
        return none_default_text_input("Search ID")

    def cursor():
        return none_default_text_input("Cursor")

    def kaggle_dataset():
        st.markdown("Visit [kaggle.com/datasets](https://www.kaggle.com/datasets)")
        st.markdown("---------------")
        return none_default_text_input("Kaggle dataset url")

    def huggingface_dataset():
        st.markdown("Visit [huggingface.co/datasets](https://huggingface.co/datasets)")
        st.markdown("---------------")
        return none_default_text_input("Huggingface dataset url")

    def delete_temp_data():
        return st.checkbox("Delete unsaved preview data afterwards")

    options = {
        "keywords": keywords_and,
        "keywordsOR": keywords_or,
        "count": count,
        "images": images,
        "start_date": start_date,
        "end_date": end_date,
        "locations": locations,
        "hashtags": hashtags,
        "video_url": video_url,
        "search_id": search_id,
        "cursor": cursor,
        "kaggle_dataset": kaggle_dataset,
        "huggingface_dataset": huggingface_dataset,
        "delete_temp_data": delete_temp_data,
    }

    with container:
        try:
            return options[option]()
        except KeyError:
            raise ValueError(f"Unknown query option: {option}")
