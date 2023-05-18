import streamlit as st

from data_collection import preview_tweets, save_tweets

title = "Data Collection"

st.set_page_config(page_title=title)

st.sidebar.header(title)

option = st.sidebar.multiselect("Social Medias", ["Twitter", "Reddit (Coming Soon)"])
keywords = st.sidebar.text_input("Enter keywords:")

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
        start, end, tweet_count, tweets = preview_tweets(keywords)
        st.session_state.start = start
        st.session_state.end = end
        st.session_state.tweet_count = tweet_count
        st.session_state.tweets = tweets
        st.success('Tweets preview are now available. Click on the "Save" button below to store all the tweets.',
                   icon="✅")

if st.session_state.tweet_count != 0:
    st.text(f"There is {st.session_state.tweet_count} tweets from {st.session_state.start} to {st.session_state.end}")
    st.dataframe(st.session_state.tweets)
    if st.button("Save"):
        save_tweets(keywords, 300)
        st.success(f"Tweets now saved as \"data/{keywords}.csv\"")
