"""
Scrapes from Facebook using the facebook-scraper package

"""

import streamlit as st
from facebook_scraper import get_posts_by_search

from ..utils import BaseDataCollector, mike, save_data

credentials = (mike["gmail_username"], mike["facebook_password"])


options = {"progress": True}


def format_posts(posts):
    formatted_posts = []
    for p in posts:
        id = p["post_id"] if "post_id" in p else None
        text = p["text"] if "text" in p else None
        likes = p["likes"] if "likes" in p else None
        comments = p["comments"] if "comments" in p else None
        shares = p["shares"] if "shares" in p else None
        time = p["time"] if "time" in p else None
        image = p["image"] if "image" in p else None
        video = p["video"] if "video" in p else None
        username = p["username"] if "username" in p else None
        user_id = p["user_id"] if "user_id" in p else None
        post_url = p["post_url"] if "post_url" in p else None

        formatted_posts.append(
            {
                "id": id,
                "text": text,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "time": time,
                "image_urls": [image],
                "video": video,
                "username": username,
                "user_id": user_id,
                "post_url": post_url,
            }
        )
    return formatted_posts


def grab_posts(query, max_results):
    posts = []
    with st.spinner("Collecting Facebook posts"):
        progress = st.text("Setting up scraper")
        for post in get_posts_by_search(query, credentials=credentials, options=options, pages=10):
            posts.append(post)
            progress.progress(len(posts) / max_results, f"{len(posts)} / {max_results} Facebook posts collected")
            if len(posts) >= max_results:
                break

    progress.empty()
    posts = format_posts(posts)
    return posts


if __name__ == "__main__":
    """
    For testing purposes
    """
    posts = grab_posts("keyboard", 5)

    posts = format_posts(posts)
    print(len(posts), "posts Saved to", save_data(posts, "facebook"))


class Collector(BaseDataCollector):
    def __init__(self):
        pass

    def query_options(self):
        return ["count", "keywords"]

    def collect(self, keywords, count):
        """
        Collects posts from Facebook via a facebook-scraper
        :param keywords: A list of keywords to search for
        :param count: The number of posts to collect
        :return: A list of post dictionaries
        """
        return grab_posts(keywords, count)
