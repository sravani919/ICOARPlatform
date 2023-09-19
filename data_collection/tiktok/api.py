import datetime
import json

import requests
import streamlit as st

from data_collection.utils import BaseDataCollector


def video_response_parsing(response) -> (str, dict):
    """
    Parses the response from the TikTok API
    :param response: The response from the TikTok API, should be a json object
    :return: A list of dictionaries containing the formatted video data
             Will return None as results if an error occurred
             The search_id is needed to resume a previous search
    """
    try:
        data = response["data"]
        has_more = data["has_more"]
    except KeyError:
        st.error(json.dumps(response["error"], indent=4))
        return None, None

    try:
        search_id = data["search_id"]
    except KeyError:
        search_id = None

    if not has_more:
        search_id = None  # No more videos to get, the search_id already should be None but just in case

    formatted_videos = []
    for video in data["videos"]:
        formatted_videos.append(
            {
                "id": video["id"],
                "text": video["video_description"],
                "create_time": video["create_time"],
                "region_code": video["region_code"],
                "username": video["username"],
                "like_count": video.get("like_count", None),
                "comment_count": video.get("comment_count", None),
                "share_count": video.get("share_count", None),
                "view_count": video.get("view_count", None),
                "music_id": video.get("music_id", None),
                "hashtag_names": video["hashtag_names"],
                "effect_ids": video.get("effect_ids", None),
                "playlist_id": video.get("playlist_id", None),
                "voice_to_text": video.get("voice_to_text", None),
                "video_url": f"https://www.tiktok.com/@{video['username']}/video/{video['id']}",
            }
        )

    return search_id, formatted_videos


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
        start_date,
        end_date,
        locations,
        hashtags,
    ) -> list[dict]:
        """
        Grabs videos from the TikTokApi that match the given parameters.
        :param hashtags: Hashtags that the video must have e.g. ["funny", "comedy"]
        :param start_date: Earliest creation date of the videos e.g. "20220615" i.e. yyyymmdd
        :param end_date: Latest creation date of the videos e.g. "20220628" i.e. yyyymmdd
        :param max_count: Maximum number of videos to return
        :param keywords: Keywords that must be in the video description
        :param locations: Locations that the videos must be from e.g. ["US", "CA"]
        :return: List of dictionaries containing the videos' data
        """

        base_data = {
            "query": {"and": []},
        }

        headers = {"authorization": f"bearer {self.access_token}"}

        if keywords is not None:
            for keyword in keywords:
                base_data["query"]["and"].append(
                    {"operation": "EQ", "field_name": "keyword", "field_values": [keyword]}
                )

        if hashtags is not None:
            for hashtag in hashtags:
                base_data["query"]["and"].append(
                    {"operation": "EQ", "field_name": "hashtag_name", "field_values": [hashtag]}
                )

        if locations is not None:
            base_data["query"]["and"].append(
                {"operation": "IN", "field_name": "region_code", "field_values": locations}
            )

        base_data["start_date"] = start_date
        base_data["end_date"] = end_date

        count_still_needed = max_count
        search_id = None

        collected_videos = []

        while count_still_needed > 0:
            data = base_data.copy()
            count_for_this_request = min(count_still_needed, 100)

            # Adding more fields to the data
            data["max_count"] = count_for_this_request
            if search_id is not None:
                data["search_id"] = search_id  # To resume a previous search

            r = requests.post(
                f"https://open.tiktokapis.com/v2/research/video/query/?fields={TikTokApi.all_fields}",
                headers=headers,
                json=data,
            )
            search_id, results = video_response_parsing(r.json())

            if results is None:
                break  # Error occurred

            collected_videos += results
            count_still_needed = max_count - len(collected_videos)

            if search_id is None:
                break

        return collected_videos


def cant_find_keys():
    with st.container():
        st.error("Could not find TikTok API credentials. Please add them to your secrets.toml file.")
        st.markdown(
            """
        Example TikTok secrets.toml

            [tiktok]
            client_key = 'KEY'
            client_secret = 'SECRET'"""
        )


def get_videos(
    keywords: str = None,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    locations: str = None,
    count: int = 100,
    hashtags: str = None,
):
    """
    Uses the TikTok API to get videos
    Helps to format the query request and check for credentials
    Designed to make interfacing with the tiktokapi easier.
    Will look for credentials inside streamlit's secrets.toml file
    :param keywords: A single string  with comma seperated keywords
    :param start_date: A datetime.date object
    :param end_date: A datetime.date object must be after start_date and within 30 days
    :param locations: Comma seperated country abbreviations
    :param count: The number of videos to return
    :param hashtags: A single string with comma seperated hashtags
    :return:
    """
    if keywords is None and hashtags is None and locations is None:
        st.error("Please specify at least one of the following: keywords, hashtags, locations")

    if type(start_date) == datetime.date:
        start_date = start_date.strftime("%Y%m%d")
    if type(end_date) == datetime.date:
        end_date = end_date.strftime("%Y%m%d")

    if keywords is not None and keywords != "":
        keywords = keywords.split(",")
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
        client_key = st.secrets.tiktok.client_key
        client_secret = st.secrets.tiktok.client_secret
    except AttributeError:
        cant_find_keys()
        return

    ttapi = TikTokApi(client_key, client_secret)
    return ttapi.video_request(count, keywords, start_date, end_date, locations, hashtags)


class Collector(BaseDataCollector):
    def __init__(self):
        super().__init__()

    def query_options(self):
        return ["count", "keywords", "start_date", "end_date", "locations", "hashtags"]

    def collect(self, **kwargs):
        return get_videos(**kwargs)
