from datetime import datetime
from math import inf

import pandas as pd
import streamlit as st
import tweepy


def init_connection():
    return tweepy.Client(st.secrets.api_token.twitter)


@st.cache_data
def preview_tweets(keywords):
    client = init_connection()
    res = client.get_recent_tweets_count(keywords)
    start = min([datetime.strptime(data["start"], "%Y-%m-%dT%H:%M:%S.%fZ") for data in getattr(res, "data")])
    end = max([datetime.strptime(data["end"], "%Y-%m-%dT%H:%M:%S.%fZ") for data in getattr(res, "data")])
    tweet_count = sum([data["tweet_count"] for data in getattr(res, "data")])
    res = client.search_recent_tweets(keywords)
    tweets = [{"id": tweet.data["id"], "text": tweet.data["text"]} for tweet in getattr(res, "data")]
    return start, end, tweet_count, tweets


def save_tweets(keywords, limit=inf):
    client = init_connection()
    tweets = []
    for tweet in tweepy.Paginator(client.search_recent_tweets, keywords, max_results=100).flatten(limit=limit):
        tweets.append({"id": tweet.id, "text": tweet.text})
    df = pd.DataFrame(tweets)
    df.to_csv(f"data/{keywords}.csv", index=False)
