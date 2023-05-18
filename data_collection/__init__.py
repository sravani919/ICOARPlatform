from datetime import datetime
from math import inf

import pandas as pd
import streamlit as st
import tweepy


def init_connection():
    return tweepy.Client(st.secrets.api_token.twitter)


@st.cache_data
def preview_tweets(keywords):
    progress_bar = st.progress(0, text="Establishing the connection with Twitter. Please wait...")

    client = init_connection()

    progress_bar.progress(10, text="Fetching tweets metadata. Please wait...")

    res = client.get_all_tweets_count(keywords)
    start = min([datetime.strptime(data["start"], "%Y-%m-%dT%H:%M:%S.%fZ") for data in getattr(res, "data")])
    end = max([datetime.strptime(data["end"], "%Y-%m-%dT%H:%M:%S.%fZ") for data in getattr(res, "data")])

    tweet_count = sum([data["tweet_count"] for data in getattr(res, "data")])

    progress_bar.progress(50, text="Fetching preview tweets. Please wait...")

    res = client.search_recent_tweets(keywords)

    progress_bar.progress(90, text="Processing preview tweets. Please wait...")

    tweets = [{"id": tweet.data["id"], "text": tweet.data["text"]} for tweet in getattr(res, "data")]

    progress_bar.empty()

    return start, end, tweet_count, tweets


def save_tweets(keywords, limit=inf):
    progress_bar = st.progress(0, text="Establishing the connection with Twitter. Please wait...")
    client = init_connection()

    progress_bar.progress(10, text="Fetching tweets. Please wait...")
    tweets = []
    for i, tweet in enumerate(tweepy.Paginator(client.search_recent_tweets, keywords, max_results=100).flatten(limit=limit)):
        tweets.append({"id": tweet.id, "text": tweet.text})
        progress = int((i+1) / limit * 100)
        progress_bar.progress(progress, text=f"Fetching tweets: {progress}% complete")

    progress_bar.empty()

    df = pd.DataFrame(tweets)
    df.to_csv(f"data/{keywords}.csv", index=False)
