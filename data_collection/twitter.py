import os
from datetime import datetime
from urllib.request import HTTPCookieProcessor, build_opener

import pandas as pd
import streamlit as st
import tweepy
from PIL import Image


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
            "attachments.media_keys",
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
        media_fields=[
            "alt_text",
            "duration_ms",
            "height",
            "media_key",
            "non_public_metrics",
            "organic_metrics",
            "preview_image_url",
            "promoted_metrics",
            "public_metrics",
            "type",
            "url",
            "variants",
            "width",
        ],
    )
    return res


def get_urls(tweet, all_media):
    """
    Gets the image urls from the tweet
    Each tweet has a list of media keys, those keys can be used to search the media for the correct urls
    :param tweet: The tweet with images we want to get the urls for
    :param all_media: The media from the response from the Twitter API (Sections are removed as they are searched)
    :return: A list of urls to the images in the tweet
    """
    try:
        tweets_media_keys = tweet["attachments"]["media_keys"]  # The media keys associated with this tweet
    except KeyError:
        return []  # If there are no media keys, there are no images in the tweet
    except TypeError:
        return []  # Couldn't find the attachments key

    media_urls = []
    # Iterating through all the media to see if any of them match the media keys from the tweet
    # If they do match, that media is removed from the all_media list so future searches are faster
    for media in all_media.copy():
        if media["media_key"] in tweets_media_keys:
            url = media["url"]
            if url is not None:
                media_urls.append(media["url"])
            all_media.remove(media)

    return media_urls


def format_tweet_results(res):
    """
    Converts the JSON response from Twitter into a list of dictionaries with the data we want
    :param res: The JSON response from Twitter
    :param download_images: Whether to download the images from the tweets or just save their links
    :return: A list of dictionaries with the data we want
    """
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
        image_urls = []

        if res.data[i].entities is not None:  # When there are no entities, there are no hashtags, mentions, or urls
            # Getting hashtags
            if "hashtags" in res.data[i].entities:
                for hashtag in res.data[i].entities["hashtags"]:
                    hashtags.append(hashtag["tag"])

            # Getting mentions
            if "mentions" in res.data[i].entities:
                for mention in res.data[i].entities["mentions"]:
                    mentions.append(mention["username"])

            # Getting direct image urls
            image_urls = get_urls(res.data[i], res.includes["media"])

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
                "image_urls": image_urls,
            }
        )
    return tweets


@st.cache_data
def grab_tweets(keywords, count: int):
    client = init_connection()
    res = client.get_recent_tweets_count(keywords)
    start = min([datetime.strptime(data["start"], "%Y-%m-%dT%H:%M:%S.%fZ") for data in getattr(res, "data")])
    end = max([datetime.strptime(data["end"], "%Y-%m-%dT%H:%M:%S.%fZ") for data in getattr(res, "data")])
    tweet_count = sum([data["tweet_count"] for data in getattr(res, "data")])

    # Getting the tweets
    res = full_results(client, keywords, count)
    if res.data is None:  # No tweets found with the given keywords
        return start, end, tweet_count, []

    # Formatting the tweets into a list of dictionaries
    tweets = format_tweet_results(res)

    return start, end, tweet_count, tweets


def save_images(tweet, images_path):
    """
    Takes in a tweet dictionary and then downloads the images off the internet and saves them to the given file path
    :param tweet: The tweet dictionary
    :param images_path: The file path to save the images to

    """
    id = tweet["id"]
    tweet_folder = images_path + "/" + str(id)  # Folder for all of the images from this tweet
    # Creating a folder for the images if it doesn't already exist
    if not os.path.exists(tweet_folder):
        os.makedirs(tweet_folder)

    urls = tweet["image_urls"]
    if urls is None or len(urls) == 0:
        return []

    opener = build_opener(HTTPCookieProcessor())

    for i, url in enumerate(urls):
        if url is None:
            continue

        response = opener.open(url)
        image = Image.open(response).convert("RGB")
        image.save(f"{tweet_folder}/{i}.jpg")


def save_tweets(tweets, filename, download_images=False):
    # client = init_connection()
    # tweets = []
    # for tweet in tweepy.Paginator(client.search_recent_tweets, keywords, max_results=100).flatten(limit=limit):
    #     tweets.append({"id": tweet.id, "text": tweet.text})
    df = pd.DataFrame(tweets)
    file_path = f"data/{filename}.csv"
    df.to_csv(file_path, index=False)

    images_path = None
    if download_images:
        images_path = f"data/{filename}_images"
        if not os.path.exists(images_path):
            os.makedirs(images_path)

        # Iterate through the tweets and download the images
        for tweet in tweets:
            if len(tweet["image_urls"]) > 0:
                save_images(tweet, images_path)

    return file_path, images_path
