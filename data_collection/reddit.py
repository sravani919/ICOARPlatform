import os
from datetime import datetime

import praw
import requests
import streamlit as st


def init_connection():
    reddit = praw.Reddit(
        client_id="99OTDRHSAKWRJxGTgdr9tw",
        client_secret="FbJfe84kzDsTC2UKBD6RO_DSPM6vhQ",
        user_agent="Character_Growth1181",
    )
    return reddit


def fetch_data(reddit, keywords, max_results, collect_images, only_images):
    subreddit = reddit.subreddit("all")
    results = subreddit.search(keywords, sort="new", limit=max_results)

    data = []
    for post in results:
        if only_images and (post.is_self or not post.url.endswith((".jpg", ".jpeg", ".png", ".gif"))):
            continue

        if collect_images and not post.is_self and post.url.endswith((".jpg", ".jpeg", ".png", ".gif")):
            image_urls = [post.url]
        else:
            image_urls = []

        data.append(
            {
                "id": post.id,
                "title": post.title,
                "author": post.author.name,
                "score": post.score,
                "post_url": f"https://www.reddit.com{post.permalink}",
                "created_utc": datetime.utcfromtimestamp(post.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                "num_comments": post.num_comments,
                # changed from selftext to text to better align with the preprocessing
                "text": post.selftext,
                "total_awards_received": post.total_awards_received,
                "over_18": post.over_18,
                "image_urls": image_urls,
            }
        )

    return data


def download_images(posts, filename, i):
    """
    Downloads the images from the given posts to the given file path
    :param posts: The list of post dictionaries
    :param filename: The filename being used to save the posts
    :param i: The index of the post to download the images from
    """
    post = posts[i]
    images_path = ""

    for i, url in enumerate(post["image_urls"]):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                file_name = f"{post['id']}_{i}.jpg"
                images_path = f"data/{filename}_images"
                if not os.path.exists(images_path):
                    os.makedirs(images_path)
                file_path = os.path.join(images_path, file_name)
                with open(file_path, "wb") as file:
                    file.write(response.content)
        except requests.exceptions.RequestException:
            continue

    return images_path


@st.cache_data
def grab_posts(keywords, tweet_count, must_have_images):
    reddit = init_connection()

    collect_images = True

    posts = fetch_data(reddit, keywords, tweet_count, collect_images, must_have_images)

    if collect_images:
        for i, post in enumerate(posts):
            download_images(posts, "images", i)

    # df = pd.DataFrame(posts)
    # # st.dataframe(df)

    return "", "", tweet_count, posts
