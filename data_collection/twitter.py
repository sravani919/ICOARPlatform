from datetime import datetime

import pandas as pd
import streamlit as st
import tweepy


def init_connection():
    return tweepy.Client(st.secrets.api_token.twitter)


def full_results(client, keywords, max_results):
    """
    Uses extensions and fields to get a lot more data from Twitter than the default.
    :param client: The tweepy.Client made from init_connection()
    :param keywords: The keywords to search for
    :param max_results: The maximum number of results to return
    :return: The full results from the search organized as a list of dictionaries
    """
    res = client.search_recent_tweets(
        keywords,
        max_results=max_results,
        expansions=[
            "author_id",
            "referenced_tweets.id",
            "referenced_tweets.id.author_id",
            "entities.mentions.username,geo.place_id",
        ],
        tweet_fields=[
            "attachments",
            "author_id",
            "context_annotations",
            "conversation_id",
            "created_at",
            "edit_controls",
            "edit_history_tweet_ids",
            "entities",
            "geo",
            "id",
            "in_reply_to_user_id",
            "lang",
            "possibly_sensitive",
            "public_metrics",
            "referenced_tweets",
            "reply_settings",
            "source",
            "text",
        ],
        user_fields=[
            "created_at",
            "description",
            "entities",
            "id",
            "location",
            "name",
            "pinned_tweet_id",
            "profile_image_url",
            "protected",
            "public_metrics",
            "url",
            "username",
            "verified",
            "verified_type",
            "withheld",
        ],
        place_fields=["contained_within", "country", "country_code", "full_name", "geo", "id", "name", "place_type"],
    )
    return res


def format_tweet_results(res):
    tweets = []
    for i in range(len(res.data)):
        id = res.data[i].id
        text = res.data[i].text

        author_id = res.includes["users"][i]["id"]
        author_name = res.includes["users"][i]["name"]
        author_username = res.includes["users"][i]["username"]

        # Public Metrics
        # retweet_count, reply_count, like_count, quote_count, impression_count
        # Only retweet_count has a large variety of values, the rest are almost always 0.
        retweet_count = res.data[i].public_metrics["retweet_count"]

        hashtags = []
        mentions = []

        if res.data[i].entities is not None:  # Is None when there are no hashtags or mentions
            # Getting hashtags
            if "hashtags" in res.data[i].entities:
                for hashtag in res.data[i].entities["hashtags"]:
                    hashtags.append(hashtag["tag"])

            # Getting mentions
            if "mentions" in res.data[i].entities:
                for mention in res.data[i].entities["mentions"]:
                    mentions.append(mention["username"])

        created_at = res.data[i].created_at

        tweets.append(
            {
                "id": id,
                "text": text,
                "date": created_at,
                "author_id": author_id,
                "author_name": author_name,
                "author_username": author_username,
                "retweet_count": retweet_count,
                "hashtags": hashtags,
                "mentions": mentions,
            }
        )
    return tweets


@st.cache_data
def preview_tweets(keywords, count):
    client = init_connection()
    res = client.get_recent_tweets_count(keywords)
    start = min([datetime.strptime(data["start"], "%Y-%m-%dT%H:%M:%S.%fZ") for data in getattr(res, "data")])
    end = max([datetime.strptime(data["end"], "%Y-%m-%dT%H:%M:%S.%fZ") for data in getattr(res, "data")])
    tweet_count = sum([data["tweet_count"] for data in getattr(res, "data")])

    # Getting the tweets
    res = full_results(client, keywords, count)
    if res.data is None:  # No tweets found with the given keywords
        return start, end, tweet_count, []

    tweets = format_tweet_results(res)

    return start, end, tweet_count, tweets


def save_tweets(tweets, filename):
    # client = init_connection()
    # tweets = []
    # for tweet in tweepy.Paginator(client.search_recent_tweets, keywords, max_results=100).flatten(limit=limit):
    #     tweets.append({"id": tweet.id, "text": tweet.text})
    df = pd.DataFrame(tweets)
    file_path = f"data/{filename}.csv"
    df.to_csv(file_path, index=False)
    return file_path
