import datetime
import json

import requests
import streamlit as st


def video_response_parsing(response) -> (str, dict):
    try:
        data = response["data"]
    except KeyError:
        print("Error: " + json.dumps(response["error"], indent=4))
        return None, None

    try:
        search_id = data["search_id"]
    except KeyError:
        search_id = None

    if not data["has_more"]:
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
        keywords: list[str],
        start_date: str,
        end_date: str,
        locations: list[str] = None,
        hashtags: list[str] = None,
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

        for keyword in keywords:
            base_data["query"]["and"].append({"operation": "EQ", "field_name": "keyword", "field_values": [keyword]})

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
            print("Amount still needed: ", count_still_needed)
            data = base_data.copy()
            count_for_this_request = min(count_still_needed, 100)

            # Adding more fields to the data
            data["max_count"] = count_for_this_request
            if search_id is not None:
                data["search_id"] = search_id  # To resume a previous search

            print("data: ", json.dumps(data, indent=4))

            r = requests.post(
                f"https://open.tiktokapis.com/v2/research/video/query/?fields={TikTokApi.all_fields}",
                headers=headers,
                json=data,
            )
            search_id, results = video_response_parsing(r.json())
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
    start_date: str = None,
    end_date: str = None,
    locations: list[str] = None,
    max_count: int = 100,
    hashtags: str = None,
):
    if keywords is not None and keywords != "":
        keywords = keywords.split(",")
    if hashtags is not None and hashtags != "":
        hashtags = hashtags.split(",")

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
    return ttapi.video_request(max_count, keywords, start_date, end_date, locations, hashtags)
