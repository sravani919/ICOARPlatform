"""
Built to work using Russell-Newton's TikTok Scraper library: https://github.com/Russell-Newton/TikTokPy

Can scrape about 2 videos per second
"""
from tiktokapipy.async_api import TikTokAPI
from tqdm.auto import tqdm

from data_collection.utils import BaseDataCollector, save_data


def format_videos(videos):
    """
    Formats the given list of videos into a list of dictionaries
    :param videos: A list of videos as returned from TikTokAPI
    :return: A list of dictionaries containing the formatted video data
    """

    formatted_videos = []
    for video in videos:
        formatted_videos.append(
            {
                "id": video.id,
                "text": video.desc,
                "author:": video.author.unique_id,
                "likes": video.stats.digg_count,
                "shares": video.stats.share_count,
                "comments": video.stats.comment_count,
                "plays": video.stats.play_count,
                "created_at": video.create_time.timestamp(),
                "video_url": video.video.download_addr,
                "thumbnail_url": video.video.cover,
                "music": video.music.title,
            }
        )
    return formatted_videos


def hashtag_search(hashtag, max_results):
    """
    Collects the first max_results number of videos with the given hashtag/challenge
    TikTok refers to hashtags as challenges internally
    :param hashtag: The hashtag/challenge to search for e.g. #fun
    :param max_results: The maximum number of videos to collect
    :return: A list of video dictionaries
    """

    if hashtag[0] == "#":
        hashtag = hashtag[1:]

    with TikTokAPI() as api:
        challenge = api.challenge(hashtag, video_limit=max_results)
        videos = []

        loading_bar = tqdm(total=max_results, desc="Collecting videos", position=0, leave=True)
        for video in challenge.videos:
            loading_bar.update(1)
            videos.append(video)
            if len(videos) >= max_results:
                break

    return format_videos(videos)


def get_video(video_id):
    """
    Collects the video with the given id
    Is only useful if you already have the id of the video you want
    :param video_id: The id of the video to collect
    :return: A list containing the video dictionary
    """

    with TikTokAPI() as api:
        video = api.video(video_id)

    return format_videos([video])


if __name__ == "__main__":
    """
    Variables to change for testing below
    """

    hashtag = "chicken"
    count = 50

    """
    Variables to change for testing above
    """

    vids = hashtag_search(hashtag, count)
    print("tiktoks saved to ", save_data(vids, hashtag, folder_path="../../data"))


class Collector(BaseDataCollector):
    def __init__(self):
        pass

    def query_options(self):
        return ["count", "hashtag"]

    def collect(self, hashtag, count):
        return hashtag_search(hashtag, count)
