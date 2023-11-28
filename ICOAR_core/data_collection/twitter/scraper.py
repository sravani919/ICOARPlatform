import asyncio
import toml
from queue import Queue

from twscrape import API, AccountsPool, gather

from ..utils import BaseDataCollector, ProgressUpdate


def get_accounts():
    # Reading from accounts.csv where the info is seperated by colons, not commas
    secrets = toml.load(".streamlit/secrets.toml")
    f = secrets["twitter"]["accounts"]

    accounts = []
    for account in f.split(","):
        values = [v.strip() for v in account.split(":")]
        accounts.append(values)
    return accounts


def format_tweets(tweets):
    formatted_tweets = []
    for tweet in tweets:
        mentions = []
        for mention in tweet.mentionedUsers:
            mentions.append(mention.username)

        coordinates = None
        if tweet.coordinates is not None:
            coordinates = [tweet.coordinates.longitude, tweet.coordinates.latitude]

        links = []
        for link in tweet.links:
            links.append(link.url)

        image_urls = []
        for photo in tweet.media.photos:
            image_urls.append(photo.url)
        # for video in tweet.media.videos:
        #     video_urls.append(video.thumbnailUrl)

        links = []
        for link in tweet.links:
            links.append(link.url)

        place = tweet.place.fullName if tweet.place is not None else None

        formatted_tweet = {
            "id": tweet.id,
            "text": tweet.rawContent,
            "date": tweet.date,
            "author_id": tweet.user.id,
            "author_name": tweet.user.username,
            "retweets": tweet.retweetCount,
            "lang": tweet.lang,
            "mentions": mentions,
            "hashtags": tweet.hashtags,
            "coordinates": coordinates,
            "place": place,
            "url": tweet.url,
            "image_urls": image_urls,
            "links": links,
        }
        formatted_tweets.append(formatted_tweet)
    return formatted_tweets


async def a_grab_tweets(keywords, tweet_count, must_have_images, start, end):
    # Setting up the accounts pool
    pool = AccountsPool()
    for account in get_accounts():
        await pool.add_account(*account)

    # Logging in to all new accounts
    # yield ProgressUpdate(0, "Logging in to Twitter accounts...")
    await pool.login_all()

    # Creating an API object
    api = API(pool)

    # Search API (latest tab)
    query = keywords[:]

    if start is not None:
        query += f" since:{start}"
    if end is not None:
        query += f" until:{end}"
    if must_have_images:
        query += " filter:images"

    # yield ProgressUpdate(0, "Searching for tweets...")
    generator = api.search(query, limit=tweet_count)
    tweets = await gather(generator)
    # yield ProgressUpdate(1, "Formatting tweets...")

    formatted_tweets = format_tweets(tweets)
    # yield formatted_tweets
    return formatted_tweets


def grab_posts(keywords, count, images, start_date, end_date):
    result = asyncio.run(a_grab_tweets(keywords, count, images, start_date, end_date))
    return result


class Collector(BaseDataCollector):
    def __init__(self):
        pass

    def query_options(self):
        return ["count", "keywords", "images", "start_date", "end_date"]

    def collect_generator(self, keywords, count, images, start_date, end_date):
        yield grab_posts(keywords, count, images, start_date, end_date)
