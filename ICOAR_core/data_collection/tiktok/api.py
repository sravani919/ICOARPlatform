import datetime
import json
import logging
import time
import toml

import requests

from ..utils import BaseDataCollector, ProgressUpdate


def video_response_parsing(response) -> (str, dict):
    """
    Parses the response from the TikTok API
    :param response: The response from the TikTok API, should be a json object
    :return: A list of dictionaries containing the formatted video data
             Will return None as results if an error occurred
             The cursor is needed to resume a previous search
             The search_id is needed to resume a previous search
    """
    try:
        data = response["data"]
        has_more = data["has_more"]
    except KeyError:
        logging.error("No data in response, message from TikTok: " + json.dumps(response["error"], indent=4))

        return None, None, None

    try:
        search_id = data["search_id"]
    except KeyError:
        search_id = None

    try:
        cursor = data["cursor"]
    except KeyError:
        cursor = None

    if not has_more:
        search_id = None  # No more videos to get, the search_id already should be None but just in case
        cursor = None

    formatted_videos = []
    for video in data["videos"]:
        # Calculating the date from the time stamp which is in seconds
        create_date = datetime.datetime.fromtimestamp(video["create_time"]).strftime("%Y%m%d")
        username = video.get("username", None)
        formatted_videos.append(
            {
                "id": video.get("id", None),
                "text": video.get("video_description", None),
                "create_time": video["create_time"],
                "create_date": video.get("create_date", create_date),
                "region_code": video.get("region_code", None),
                "username": username,
                "like_count": video.get("like_count", None),
                "comment_count": video.get("comment_count", None),
                "share_count": video.get("share_count", None),
                "view_count": video.get("view_count", None),
                "music_id": video.get("music_id", None),
                "hashtag_names": video["hashtag_names"],
                "effect_ids": video.get("effect_ids", None),
                "playlist_id": video.get("playlist_id", None),
                "voice_to_text": video.get("voice_to_text", None),
                "video_url": f"https://www.tiktok.com/@{username}/video/{video['id']}",
            }
        )

    return cursor, search_id, formatted_videos


class TikTokApi:
    all_fields = (
        "id,video_description,create_time,region_code,share_count,view_count,like_count,"
        "comment_count,music_id,hashtag_names,username,effect_ids,playlist_id,voice_to_text"
    )

    def __init__(self, client_key, client_secret):
        self.client_key = client_key
        self.client_secret = client_secret
        self.access_token = self._get_access_token()

    def _get_access_token(self):
        """
        Gets the temporary access token from the TikTok API using the client key and client secret
        :return: The access token
        """
        r = requests.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            headers={"Content-Type": "application/x-www-form-urlencoded", "Cache-Control": "no-cache"},
            data={
                "client_key": self.client_key,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
            },
        )
        return r.json()["access_token"]

    def video_request(
        self,
        max_count: int,
        keywords,
        keywordsOR,
        start_date,
        end_date,
        locations,
        hashtags,
        cursor=None,
        search_id=None,
    ) -> (list[dict], str):
        """
        Grabs videos from the TikTokApi that match the given parameters.
        Generator: Will yield progress updates and then the final results
        :param hashtags: Hashtags that the video must have e.g. ["funny", "comedy"]
        :param start_date: Earliest creation date of the videos e.g. "20220615" i.e. yyyymmdd
        :param end_date: Latest creation date of the videos e.g. "20220628" i.e. yyyymmdd
        :param max_count: Maximum number of videos to return
        :param keywords: Keywords that must be in the video description
        :param keywordsOR: At least one of these keywords must be in the video description
        :param locations: Locations that the videos must be from e.g. ["US", "CA"]
        :param cursor: The cursor to resume from
        :param search_id: The search id to resume from
        :return: List of dictionaries containing the videos' data and the search_id
        """

        # print("\n\nTikTokAPI video request called with search id of", search_id)
        base_data = {
            "query": {"and": [], "or": []},
        }

        headers = {"authorization": f"bearer {self.access_token}"}

        # Adding the required keywords to the query
        if keywords is not None:
            for keyword in keywords:
                base_data["query"]["and"].append(
                    {"operation": "EQ", "field_name": "keyword", "field_values": [keyword]}
                )

        # At least one of the keywords must be in the video description
        # Adding them to the request
        if keywordsOR is not None:
            for keyword in keywordsOR:
                base_data["query"]["or"].append({"operation": "EQ", "field_name": "keyword", "field_values": [keyword]})

        # Adding the required hashtags to the query
        if hashtags is not None:
            for hashtag in hashtags:
                base_data["query"]["and"].append(
                    {"operation": "EQ", "field_name": "hashtag_name", "field_values": [hashtag]}
                )

        # Adding the required locations to the query
        if locations is not None:
            base_data["query"]["and"].append(
                {"operation": "IN", "field_name": "region_code", "field_values": locations}
            )

        # Setting the date range
        base_data["start_date"] = start_date
        base_data["end_date"] = end_date

        count_still_needed = max_count

        collected_videos = []

        no_data_error_count = 0  # The number of sequential fails to get data
        start_time = time.time()
        while count_still_needed > 0 and no_data_error_count < 5:
            estimated_seconds_left = (time.time() - start_time) / (len(collected_videos) + 1) * count_still_needed
            # Creating a string with the estimated time left as minutes and seconds
            estimated_time_left = (
                f"{int(estimated_seconds_left // 60)} minutes {int(estimated_seconds_left % 60)} seconds"
            )
            yield ProgressUpdate(
                (max_count - count_still_needed) / max_count,
                f"Collecting... {estimated_time_left} left | {len(collected_videos)} collected",
            )

            data = base_data.copy()
            count_for_this_request = 100  # Always ask for 100 videos, we are limited by requests not by videos

            # Adding more fields to the data
            data["max_count"] = count_for_this_request
            if cursor is not None:
                data["cursor"] = cursor  # To resume a previous search
            if search_id is not None:
                data["search_id"] = search_id

            # print("data in request that is being sent to tiktok:", data)

            r = requests.post(
                f"https://open.tiktokapis.com/v2/research/video/query/?fields={TikTokApi.all_fields}",
                headers=headers,
                json=data,
            )

            # print("search id variable value before updating it:", search_id)
            results = None
            try:
                t_cursor, t_search_id, results = video_response_parsing(r.json())
            except json.decoder.JSONDecodeError:
                yield ProgressUpdate(0, "Error decoding json from TikTok API")
                logging.error("Error decoding json from TikTok API")
            except KeyError:
                yield ProgressUpdate(0, "Error parsing response from TikTok API")
                logging.error("Error parsing response from TikTok API")

            if results is None:
                # If we got no data, increment the counter and try again with the same cursor and search_id
                no_data_error_count += 1
                logging.warning(f"No data in response from TikTok API -- {no_data_error_count} / 5")
            else:
                # If we got valid data, reset the counter and update the cursor and search_id
                no_data_error_count = 0
                cursor = t_cursor
                search_id = t_search_id
                collected_videos += results
                count_still_needed = max_count - len(collected_videos)

            if cursor is None:
                break

        yield collected_videos, {"cursor": cursor, "search_id": search_id}


def cant_find_keys():
    return """
Could not find TikTok API credentials. Please add them to your secrets.toml file.
Example TikTok secrets.toml:

[tiktok]
client_key = 'KEY'
client_secret = 'SECRET'
"""


def get_videos(
    keywords: str = None,
    keywordsOR: str = None,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    locations: str = None,
    count: int = 100,
    hashtags: str = None,
    cursor: int = None,
    search_id: str = None,
):
    """
    Uses the TikTok API to get videos
    Helps to format the query request and check for credentials
    Designed to make interfacing with the tiktokapi easier.
    Will look for credentials inside streamlit's secrets.toml file
    :param keywords: A single string  with comma seperated keywords
    :param keywordsOR: A single string with comma seperated keywords
    :param start_date: A datetime.date object
    :param end_date: A datetime.date object must be after start_date and within 30 days
    :param locations: Comma seperated country abbreviations
    :param count: The number of videos to return
    :param hashtags: A single string with comma seperated hashtags
    :param cursor: The cursor to resume from
    :param search_id: The search id to resume from
    :return:
    """


    """
    Preprocessing the query options, so they can be passed to the TikTok API
    """

    if keywords is None and hashtags is None and locations is None and keywordsOR is None:
        yield ProgressUpdate(0, "No keywords, hashtags, or locations specified")

    # Converting the dates to the format the TikTok API expects
    if type(start_date) == datetime.date:
        start_date = start_date.strftime("%Y%m%d")
    if type(end_date) == datetime.date:
        end_date = end_date.strftime("%Y%m%d")

    # Converting the comma seperated strings to lists
    if keywords is not None and keywords != "":
        keywords = keywords.split(",")
    if keywordsOR is not None and keywordsOR != "":
        keywordsOR = keywordsOR.split(",")
    if hashtags is not None and hashtags != "":
        hashtags = hashtags.split(",")
    if locations is not None and locations != "":
        locations = locations.split(",")

    # If start_date and end_date are not specified, default to the last 7 days
    if start_date is None or end_date is None:
        today = datetime.datetime.today()
        start_date = (today - datetime.timedelta(days=7)).strftime("%Y%m%d")
        end_date = today.strftime("%Y%m%d")

    try:
        secrets = toml.load(".streamlit/secrets.toml")
        client_key = secrets["tiktok"]["client_key"]
        client_secret = secrets["tiktok"]["client_secret"]
    except AttributeError:
        cant_find_keys()
        return

    ttapi = TikTokApi(client_key, client_secret)

    gen = ttapi.video_request(
        count, keywords, keywordsOR, start_date, end_date, locations, hashtags, cursor, search_id
    )

    for r in gen:
        # If this is a progress update, yield it
        if isinstance(r, ProgressUpdate):
            yield r
        else:
            # If this is the final results, yield it later
            results, util = r
            break

    # Sending a progress update with the final cursor and search_id
    yield ProgressUpdate(1, f"Collection complete. Final cursor: {util['cursor']}\n"
                            f"Final search_id: {util['search_id']}")
    logging.info(f"Collection complete. Final cursor: {util['cursor']}\n"
                 f"Final search_id: {util['search_id']}")
    yield results


class Collector(BaseDataCollector):
    def __init__(self):
        super().__init__()

    def query_options(self):
        return [
            "count",
            "keywords",
            "keywordsOR",
            "start_date",
            "end_date",
            "locations",
            "hashtags",
            "search_id",
            "cursor",
        ]

    def collect_generator(self, *args, **kwargs):
        yield from get_videos(*args, **kwargs)
