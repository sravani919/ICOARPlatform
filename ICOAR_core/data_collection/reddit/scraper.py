import os
from datetime import datetime

import praw

from ..utils import BaseDataCollector, ProgressUpdate, download_images


# -------------------------------------------------
# Initialize Reddit connection (LOCAL SAFE)
# -------------------------------------------------
def init_connection():
    reddit = praw.Reddit(
        client_id="99OTDRHSAKWRJxGTgdr9tw",
        client_secret="FbJfe84kzDsTC2UKBD6RO_DSPM6vhQ",
        user_agent="ICOAR_Local_Scraper",
    )
    return reddit


# -------------------------------------------------
# Fetch posts + comments
# -------------------------------------------------
def fetch_data(
    reddit,
    keywords,
    max_results,
    collect_images,
    only_images,
    get_comments,
    comment_limit,
):
    subreddit = reddit.subreddit("all")

    # ✅ CRITICAL FIX: keywords must be a string
    if isinstance(keywords, (list, tuple)):
        keywords = " OR ".join(keywords)

    yield ProgressUpdate(0.0, "Fetching posts from Reddit...")

    results = subreddit.search(
        query=keywords,
        sort="relevance",
        limit=max_results,
    )

    data = []

    for idx, post in enumerate(results):
        yield ProgressUpdate(
            min(idx / max_results, 1.0),
            f"Processing post {idx + 1}/{max_results}",
        )

        # Skip non-image posts if ONLY images requested
        if only_images and (
            post.is_self
            or not post.url.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))
        ):
            continue

        # Collect image URLs if enabled
        image_urls = []
        if collect_images and not post.is_self:
            if post.url.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                image_urls.append(post.url)

        # -----------------------------
        # Collect comments (IMPORTANT)
        # -----------------------------
        comments = []
        if get_comments:
            try:
                submission = reddit.submission(id=post.id)
                submission.comments.replace_more(limit=0)

                limit = comment_limit or 50
                for comment in submission.comments.list()[:limit]:
                    if comment.body:
                        comments.append(comment.body.replace('"', ""))
            except Exception:
                comments = []

        post_data = {
            "post_id": post.id,
            "subreddit": post.subreddit.display_name,
            "title": post.title,
            "text": post.selftext.replace('"', ""),
            "score": post.score,
            "num_comments": post.num_comments,
            "created_utc": datetime.utcfromtimestamp(
                post.created_utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "post_url": f"https://www.reddit.com{post.permalink}",
            "comments": comments,
            "image_urls": image_urls,
            "over_18": post.over_18,
        }

        data.append(post_data)

    yield data


# -------------------------------------------------
# Generator wrapper
# -------------------------------------------------
def grab_posts(keywords, count, must_have_images, get_comments, comment_limit):
    reddit = init_connection()
    collect_images = False

    posts_data = None

    for item in fetch_data(
        reddit,
        keywords,
        count,
        collect_images,
        must_have_images,
        get_comments,
        comment_limit,
    ):
        if isinstance(item, ProgressUpdate):
            yield item
        else:
            posts_data = item

    if posts_data is None:
        yield []
        return

    yield posts_data


# -------------------------------------------------
# ICOAR Collector Interface
# -------------------------------------------------
class Collector(BaseDataCollector):
    def query_options(self):
        return ["count", "keywords", "images", "get_comments", "comment_limit"]

    def auth(self) -> list[str]:
        return []

    def collect_generator(self, count, keywords, images, get_comments, comment_limit):
        yield from grab_posts(
            keywords,
            count,
            images,
            get_comments,
            comment_limit,
        )
