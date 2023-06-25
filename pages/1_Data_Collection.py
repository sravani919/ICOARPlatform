import streamlit as st

from data_collection import download_images, grab_posts, grab_tweets, save_data, save_images, save_tweets

title = "Data Collection"

st.set_page_config(page_title=title)

st.sidebar.header(title)

option = st.sidebar.multiselect("Social Medias", ["Twitter", "Reddit"])
keywords = st.sidebar.text_input("Enter keywords:")
must_have_images = st.sidebar.checkbox("Tweets must have images")
tweet_count = st.sidebar.number_input("Number of tweets:", min_value=10, max_value=100, value=100)

if "start" not in st.session_state:
    st.session_state.start = ""
if "end" not in st.session_state:
    st.session_state.end = ""
if "tweet_count" not in st.session_state:
    st.session_state.tweet_count = 0
if "tweets" not in st.session_state:
    st.session_state.tweets = []


if st.sidebar.button("Preview"):
    if option == []:
        st.sidebar.error("Please select social media")
    elif keywords == "":
        st.sidebar.error("Please enter keywords")
    else:
        if "Twitter" in option:
            start, end, tweet_count, tweets = grab_tweets(keywords, tweet_count, must_have_images)
            if not tweets:  # The list is empty
                st.sidebar.error("No tweets found with the given keywords")
            st.session_state.start = start
            st.session_state.end = end
            st.session_state.tweet_count = tweet_count
            st.session_state.tweets = tweets
            st.success(
                'Tweets preview are now available. Click on the "Save" button below to store all the tweets.', icon="✅"
            )

        if "Reddit" in option:
            start, end, tweet_count, posts = grab_posts(keywords, tweet_count, must_have_images)
            if not posts:
                st.sidebar.error("No posts found with the given keywords")
            st.session_state.start = start
            st.session_state.end = end
            st.session_state.tweet_count = tweet_count
            st.session_state.posts = posts

if st.session_state.tweet_count != 0 and st.session_state.tweets:  # Update this condition
    st.text(f"There are {st.session_state.tweet_count} tweets from {st.session_state.start} to {st.session_state.end}")
    st.text(f"Here are {len(st.session_state.tweets)} tweets")
    st.dataframe(st.session_state.tweets)

    # Having a text prompt for the name of the file to save
    filename = st.text_input("File name:", value=keywords)
    download_imagesa = st.checkbox("Download the images")
    if st.button("Save"):
        file_path = save_tweets(st.session_state.tweets, filename)
        st.success("Saved data to '" + file_path + "'")
        download_images_progress_bar = st.progress(0)
        if download_imagesa:
            image_path = ""
            for i in range(len(st.session_state.tweets)):
                image_path = save_images(st.session_state.tweets, filename, i)
                download_images_progress_bar.progress(
                    i / len(st.session_state.tweets),
                    text=f"Downloading images (images from {i+1}/{len(st.session_state.tweets)} tweets downloaded)",
                )
            st.success("Successfully downloaded all the images to '" + image_path + "'")

elif st.session_state.tweet_count != 0 and st.session_state.posts:  # Update this condition
    st.text(
        f"There are {st.session_state.tweet_count} Reddit posts from {st.session_state.start} to {st.session_state.end}"
    )
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
                image_path = download_images(st.session_state.posts, filename, i)
                download_images_progress_bar.progress(
                    i / len(st.session_state.posts),
                    text=f"Downloading images (images from {i+1}/{len(st.session_state.posts)} tweets downloaded)",
                )
            st.success("Successfully downloaded all the images to '" + image_path + "'")
