"""
Scrapes from Facebook using the facebook-scraper package

"""
import facebook_scraper
from facebook_scraper import get_posts_by_search

from ..utils import BaseDataCollector, ProgressUpdate

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


class Collector(BaseDataCollector):
    def __init__(self):
        pass

    def query_options(self):
        return ["count", "keywords"]

    def auth(self) -> list[str]:
        return ["facebook.email", "facebook.password"]

    def collect_generator(self, **kwargs):
        posts = []
        yield ProgressUpdate(0, "Starting Facebook scrape")

        credentials = (kwargs["facebook.email"], kwargs["facebook.password"])
        max_results = kwargs["count"]
        try:
            for post in get_posts_by_search(kwargs["keywords"], credentials=credentials, options=options, pages=10):
                posts.append(post)
                yield ProgressUpdate(
                    len(posts) / max_results, f"Scraping Facebook ({len(posts)}/{max_results} results scraped)"
                )
                if len(posts) >= max_results:
                    break
        except facebook_scraper.exceptions.LoginError:
            yield ProgressUpdate(1, "Login error")
            return
        posts = format_posts(posts)
        yield posts
