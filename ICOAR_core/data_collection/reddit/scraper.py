from datetime import datetime

import praw

from ..utils import BaseDataCollector, ProgressUpdate, download_images


def init_connection():
    reddit = praw.Reddit(
        client_id="99OTDRHSAKWRJxGTgdr9tw",
        client_secret="FbJfe84kzDsTC2UKBD6RO_DSPM6vhQ",
        user_agent="Character_Growth1181",
    )
    return reddit


def fetch_data(reddit, keywords, max_results, collect_images, only_images, get_comments, comment_limit):
    subreddit = reddit.subreddit("all")
    yield ProgressUpdate(0, "Fetching posts")
    results = subreddit.search(keywords, sort="relevance", limit=max_results)
    yield ProgressUpdate(0.5, "Processing posts")

    data = []
    for j, post in enumerate(results):
        yield ProgressUpdate(j / max_results, f"Processing posts ({j + 1}/{max_results} posts processed)")

        if only_images and (post.is_self or not post.url.endswith((".jpg", ".jpeg", ".png", ".gif"))):
            continue

        if collect_images and not post.is_self and post.url.endswith((".jpg", ".jpeg", ".png", ".gif")):
            image_urls = [post.url]
        else:
            image_urls = []

        if get_comments:
            submission = reddit.submission(id=post.id)
            comments = []
            submission.comments.replace_more(limit=comment_limit)
            for i, comment in enumerate(submission.comments.list()):
                yield ProgressUpdate(
                    i / comment_limit, f"Processing comments ({i + 1}/{comment_limit} comments processed)"
                )
                if i >= comment_limit:
                    break
                # comments.append(
                #     {
                #         "id": comment.id,
                #         "author": comment.author.name if comment.author else None,
                #         "score": comment.score,
                #         "created_utc": datetime.utcfromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                #         "text": comment.body,
                #     }
                # )
                # Remove double quotes
                comment.body = comment.body.replace('"', "")
                comments.append(comment.body)
        else:
            comments = []

        post_data = {
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

        for i, comment in enumerate(comments):
            post_data[f"comment_{i}"] = comment

        data.append(post_data)

    return data


def grab_posts(keywords, tweet_count, must_have_images, get_comments, comment_limit):
    reddit = init_connection()

    collect_images = False

    # Forwarding the progress updates until the data is ready
    for posts in fetch_data(
        reddit, keywords, tweet_count, collect_images, must_have_images, get_comments, comment_limit
    ):
        if isinstance(posts, ProgressUpdate):
            yield posts
            continue

    if collect_images:
        for i, post in enumerate(posts):
            yield ProgressUpdate(i / len(posts), f"Downloading images ({i + 1}/{len(posts)} images downloaded)")
            download_images(posts, "images", i)

    # df = pd.DataFrame(posts)
    # # st.dataframe(df)

    yield posts


class Collector(BaseDataCollector):
    def __init__(self):
        pass

    def query_options(self):
        return ["count", "keywords", "images", "get_comments", "comment_limit"]

    def collect_generator(self, count, keywords, images, get_comments, comment_limit):
        yield from grab_posts(keywords, count, images, get_comments, comment_limit)
