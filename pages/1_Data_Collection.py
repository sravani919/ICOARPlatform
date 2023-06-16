import streamlit as st

from data_collection import grab_tweets, save_images, save_tweets

title = "Data Collection"

st.set_page_config(page_title=title)

st.sidebar.header(title)

option = st.sidebar.multiselect("Social Medias", ["Twitter", "Reddit (Coming Soon)"])
keywords = st.sidebar.text_input("Enter keywords:")
must_have_images = st.sidebar.checkbox("Tweets must have images")
tweet_count = st.sidebar.number_input("Number of tweets:", min_value=10, max_value=10000, value=100)


if "start" not in st.session_state:
    st.session_state.start = None
if "end" not in st.session_state:
    st.session_state.end = None
if "tweet_count" not in st.session_state:
    st.session_state.tweet_count = 0
if "tweets" not in st.session_state:
    st.session_state.tweets = []

if st.sidebar.checkbox("Choose date range"):
    st.session_state.start = st.sidebar.date_input("Start date")
    st.session_state.end = st.sidebar.date_input("End date")

if st.sidebar.button("Preview"):
    if option == []:
        st.sidebar.error("Please select social media")
    elif keywords == "":
        st.sidebar.error("Please enter keywords")
    else:
        tweets = grab_tweets(keywords, tweet_count, must_have_images, st.session_state.start, st.session_state.end)
        if not tweets:  # The list is empty
            st.sidebar.error("No tweets found with the given keywords")
        st.session_state.tweet_count = tweet_count
        st.session_state.tweets = tweets
        st.success(
            'Tweets preview are now available. Click on the "Save" button below to store all the tweets.', icon="✅"
        )

if st.session_state.tweet_count != 0:
    st.text(f"Here are {len(st.session_state.tweets)} tweets")
    st.dataframe(st.session_state.tweets)

    # Having a text prompt for the name of the file to save
    filename = st.text_input("File name:", value=keywords)
    download_images = st.checkbox("Download the images")
    if st.button("Save"):
        file_path = save_tweets(st.session_state.tweets, filename)
        st.success("Saved data to '" + file_path + "'")
        download_images_progress_bar = st.progress(0)
        if download_images:
            image_path = ""
            for i in range(len(st.session_state.tweets)):
                image_path = save_images(st.session_state.tweets, filename, i)
                download_images_progress_bar.progress(
                    i / len(st.session_state.tweets),
                    text=f"Downloading images (images from {i+1}/{len(st.session_state.tweets)} tweets downloaded)",
                )
            st.success("Successfully downloaded all the images to '" + image_path + "'")
