import datetime
import time

import streamlit as st

from data_collection import reddit, tiktok, twitter
from data_collection.utils import save_data

title = "Data Collection"

st.set_page_config(page_title=title)

st.sidebar.header(title)

option = st.sidebar.selectbox("Social Medias", ["Twitter", "Reddit", "Tiktok"])
thing_names = {"Twitter": "tweets", "Reddit": "posts", "Tiktok": "tiktoks"}

thing_name = ""
if option is not None:
    thing_name = thing_names[option]
    keywords = ""
    must_have_images = False

    if not option == "Tiktok":
        keywords = st.sidebar.text_input("Enter keywords:")
        must_have_images = st.sidebar.checkbox(thing_name.capitalize() + " must have images")
        if st.sidebar.checkbox("Choose date range"):
            # The min_value is the date when Twitter was launched
            st.session_state.start = st.sidebar.date_input("Start date", min_value=datetime.date(2006, 3, 21))
            st.session_state.end = st.sidebar.date_input("End date", min_value=datetime.date(2006, 3, 21))
        else:
            st.session_state.start = None
            st.session_state.end = None
    else:
        tiktok_hashtag = st.sidebar.text_input("TikTok hashtag:")
    post_count = st.sidebar.number_input(f"Number of {thing_name}:", min_value=10, max_value=10000, value=100)

if "start" not in st.session_state:
    st.session_state.start = None
if "end" not in st.session_state:
    st.session_state.end = None
if "post_count" not in st.session_state:
    st.session_state.post_count = 0
if "tweets" not in st.session_state:
    st.session_state.tweets = []

if st.sidebar.button("Preview"):
    if option == []:
        st.sidebar.error("Please select social media")
    else:
        if "Twitter" in option:
            if keywords == "":
                st.error("Please enter keywords")

            tweets = twitter.grab_tweets(
                keywords, post_count, must_have_images, st.session_state.start, st.session_state.end
            )
            if not tweets:  # The list is empty
                st.sidebar.error("No tweets found with the given keywords")

            st.session_state.post_count = len(tweets)
            st.session_state.tweets = tweets
            if len(tweets) > 0:
                st.success(
                    f'{len(tweets)} tweets are now available. \
                    Click on the "Save" button below to store all the tweets.',
                    icon="✅",
                )

        if "Reddit" in option:
            if keywords == "":
                st.error("Please enter keywords")
            start, end, post_count, posts = reddit.grab_posts(keywords, post_count, must_have_images)
            if not posts:
                st.sidebar.error("No posts found with the given keywords")
            st.session_state.start = start
            st.session_state.end = end
            st.session_state.post_count = post_count
            st.session_state.posts = posts
            if len(posts) > 0:
                st.success(
                    f'{len(posts)} posts are now available. \
                    Click on the "Save" button below to store all the posts.',
                    icon="✅",
                )

        if "Tiktok" in option:
            if tiktok_hashtag is None or tiktok_hashtag == "":
                st.sidebar.error("Please enter a TikTok hashtag")
            else:
                tiktoks = tiktok.hashtag_search(tiktok_hashtag, post_count)

            st.session_state.post_count = len(tiktoks)
            st.session_state.posts = tiktoks

if st.session_state.post_count != 0 and st.session_state.tweets:  # Update this condition
    st.text(f"There are {st.session_state.post_count} tweets from {st.session_state.start} to {st.session_state.end}")
    st.text(f"Here are {len(st.session_state.tweets)} tweets")
    st.dataframe(st.session_state.tweets)

    # Having a text prompt for the name of the file to save
    filename = st.text_input("File name:", value=keywords)
    download_images = st.checkbox("Download the images")
    if st.button("Save"):
        file_path = save_data(st.session_state.tweets, filename)
        st.success("Saved data to '" + file_path + "'")
        download_images_progress_bar = st.empty()
        if download_images:
            image_path = ""
            pb_start = time.time()
            for i in range(len(st.session_state.tweets)):
                image_path = twitter.save_images(st.session_state.tweets, filename, i)
                time_left = (time.time() - pb_start) * (len(st.session_state.tweets) - i) / (i + 1)
                download_images_progress_bar.progress(
                    i / len(st.session_state.tweets),
                    text=f"Downloading images (images from {i+1}/{len(st.session_state.tweets)} tweets downloaded) \
                    - {time_left:.2f} seconds left",
                )
            st.success("Successfully downloaded all the images to '" + image_path + "'")

elif st.session_state.post_count != 0 and st.session_state.posts:  # Update this condition
    st.text(f"Here are {len(st.session_state.posts)} posts")
    st.dataframe(st.session_state.posts)

    # Having a text prompt for the name of the file to save
    filename = st.text_input("File name:", value=keywords)
    download_imagesa = st.checkbox("Download the images")
    if st.button("Save"):
        file_path = save_data(st.session_state.posts, filename)
        st.success("Saved data to '" + file_path + "'")
        download_images_progress_bar = st.progress(0)
        if download_imagesa:
            image_path = ""
            for i in range(len(st.session_state.posts)):
                image_path = reddit.download_images(st.session_state.posts, filename, i)
                download_images_progress_bar.progress(
                    i / len(st.session_state.posts),
                    text=f"Downloading images (images from {i+1}/{len(st.session_state.posts)} tweets downloaded)",
                )
            st.success("Successfully downloaded all the images to '" + image_path + "'")
