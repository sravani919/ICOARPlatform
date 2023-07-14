from tiktokapipy.async_api import TikTokAPI


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

    print("About to initialize TikTokAPI")

    with TikTokAPI() as api:
        print("Searching for videos with hashtag: " + hashtag)
        challenge = api.challenge(hashtag)
        videos = []
        for video in challenge.videos:
            videos.append(video)
            if len(videos) >= max_results:
                break

    return format_videos(videos)


if __name__ == "__main__":
    # for testing
    print(hashtag_search("#fun", 10))
