import os
import socket
import time  # For sleeping between requests
import urllib.error
from urllib.request import HTTPCookieProcessor, build_opener

import pandas as pd
import streamlit as st
import tweepy
from PIL import Image

IMAGE_DOWNLOAD_MAX_ATTEMPTS = 3


def init_connection():
    return tweepy.Client(st.secrets.api_token.twitter)


def full_results(client, keywords, count, must_have_images, start_time, end_time):
    """
    Uses extensions and fields to get a lot more data from Twitter than the default.
    :param start_time: The oldest date to get tweets from
    :param must_have_images: Whether every tweet must have an image attached
    :param client: The tweepy.Client made from init_connection()
    :param keywords: The keywords to search for
    :param count: The number of tweets to return
    :return: The full results from the search organized as a list of dictionaries
    """
    if start_time is not None:
        start_time = start_time.strftime("%Y-%m-%dT0:0:0Z")
    if end_time is not None:
        end_time = end_time.strftime("%Y-%m-%dT0:0:0Z")

    pb = st.progress(0)
    pb_start = time.time()

    wn = st.empty()
    wn.empty()

    query = keywords
    if must_have_images:
        query += " has:images -is:retweet"

    total_tweets_gotten = 0
    next_token = None
    res = None
    # Doing multiple requests to get all the tweets
    while total_tweets_gotten < count:
        c = count - total_tweets_gotten
        if c > 500:
            c = 500
        elif c < 10:
            c = 10

        try:
            sub_res = client.search_all_tweets(
                query,
                max_results=c,
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
                    "withheld",
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
                place_fields=[
                    "contained_within",
                    "country",
                    "country_code",
                    "full_name",
                    "geo",
                    "id",
                    "name",
                    "place_type",
                ],
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
                next_token=next_token,
                start_time=start_time,
                end_time=end_time,
            )

            # Getting the next token to avoid duplicates
            if (
                "next_token" not in sub_res.meta
                or sub_res.meta["next_token"] is None
                or sub_res is None
                or sub_res.data is None
                or len(sub_res.data) == 0
            ):
                wn.warning("No more tweets that match the search criteria")
                return res
            else:
                next_token = sub_res.meta["next_token"]

            # Appending this sub_res to the res
            if res is None:
                res = sub_res
            else:
                res.data.extend(sub_res.data)
                for key in sub_res.includes:
                    if key not in res.includes:
                        res.includes[key] = sub_res.includes[key]
                    else:
                        res.includes[key].extend(sub_res.includes[key])

            # Updating the count
            total_tweets_gotten += len(sub_res.data)

        except tweepy.errors.TooManyRequests:
            wn.warning("Too many requests, sleeping for 3 seconds")
            time.sleep(3)
            continue
        if total_tweets_gotten > count:
            total_tweets_gotten = count
        time_left = (time.time() - pb_start) / total_tweets_gotten * (count - total_tweets_gotten)
        pb.progress(
            total_tweets_gotten / count, text=f"{total_tweets_gotten}/{count} tweets gotten, {time_left:.2f}s left"
        )

    pb.empty()
    wn.empty()
    # Trimming excess tweets
    while len(res.data) > count:
        res.data.pop()

    return res


def get_image_urls(tweet, all_media):
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
        return []  # Couldn't find the attachment's keys

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
    no_media = False
    if "media" not in res.includes:
        no_media = True

    all_users = res.includes["users"].copy()

    for i in range(len(res.data)):
        id = res.data[i].id
        text = res.data[i].text

        author_id = res.data[i].author_id
        # Searching through all the users to find the author of this tweet
        author_username = None
        author_name = None
        for user in all_users:
            if user["id"] == author_id:
                author_username = user["username"]
                author_name = user["name"]
                break

        # Public Metrics
        # retweet_count, reply_count, like_count, quote_count, impression_count
        # Only retweet_count has a large variety of values; the rest are almost always 0.
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
            if not no_media:
                image_urls = get_image_urls(res.data[i], res.includes["media"])

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
                # "tweet_url": f"https://twitter.com/{author_username}/status/{id}",
            }
        )
    return tweets


@st.cache_data
def grab_tweets(keywords, count: int, must_have_images, start_time, end_time):
    client = init_connection()

    # Getting the actual tweets
    res = full_results(client, keywords, count, must_have_images, start_time, end_time)
    if res is None:
        return []
    # Formatting the tweets into a list of dictionaries
    tweets = format_tweet_results(res)

    return tweets


def download_image(url):
    """
    Downloads an image from the given url
    :param url: The direct url to the image
    :return: PIL Image object or None if the image couldn't be downloaded
    """
    if url is None:
        return

    opener = build_opener(HTTPCookieProcessor())

    attempt = 0
    while attempt < IMAGE_DOWNLOAD_MAX_ATTEMPTS:
        try:
            response = opener.open(url, timeout=2)
            image = Image.open(response).convert("RGB")
            return image
        except urllib.error.URLError:
            print(f"URLError opening {url}")
        except socket.timeout:
            print(f"Socket timeout opening {url}")
        attempt += 1
        print(f"Retrying... ({attempt}/{IMAGE_DOWNLOAD_MAX_ATTEMPTS})")

    return


def save_tweet_images(tweet, images_path):
    """
    Takes in a single tweet dictionary and then downloads the images off the internet
    and saves them to the given file path
    :param tweet: The tweet dictionary
    :param images_path: The file path to save the images to

    """
    id = tweet["id"]
    tweet_folder = images_path + "/" + str(id)  # Folder for all the images from this tweet
    # Creating a folder for the images if it doesn't already exist
    if not os.path.exists(tweet_folder):
        os.makedirs(tweet_folder)

    # Checking if the tweet has any images
    urls = tweet["image_urls"]
    if urls is None or len(urls) == 0:
        return []

    # Downloading and saving the images
    for i in range(len(urls)):
        image = download_image(urls[i])
        if image is not None:
            image.save(f"{tweet_folder}/{i}.jpg")


def save_tweets(tweets, filename):
    df = pd.DataFrame(tweets)
    file_path = f"data/{filename}.csv"
    df.to_csv(file_path, index=False)

    return file_path


def save_images(tweets, filename, i):
    """
    Saves the images from the given tweet to the given file path
    Uses an index i so a loading bar can be implemented on the front end while still having all the code here
    :param tweets: The full list of tweet dictionaries
    :param filename: The filename being used to save the tweets
    :param i: The index of the tweet to save the images from (allows for easy iteration for a loading bar)
    :return: The file path to the folder containing the images from the tweet
    """
    images_path = f"data/{filename}_images"
    if i == 0:
        if not os.path.exists(images_path):
            os.makedirs(images_path)
    save_tweet_images(tweets[i], images_path)

    return images_path
