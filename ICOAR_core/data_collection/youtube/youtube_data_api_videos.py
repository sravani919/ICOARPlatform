import googleapiclient.discovery
import toml

from ..utils import BaseDataCollector


def format_response(response):
    formatted_response = []
    for item in response:
        formatted_response.append(
            {
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "publishedAt": item["snippet"]["publishedAt"],
                "channelTitle": item["snippet"]["channelTitle"],
                "channelId": item["snippet"]["channelId"],
                "image_urls": item["snippet"]["thumbnails"]["high"]["url"],
                "id": item["id"]["videoId"],
            }
        )
    return formatted_response


class Collector(BaseDataCollector):
    def __init__(self):
        pass

    def query_options(self):
        return ["search_query", "count"]

    def auth(self) -> list[str]:
        return ["youtube.accounts"]

    def collect_generator(self, search_query, count):
        # Load in accounts from the toml
        try:
            accounts = toml.load(".streamlit/secrets.toml")["youtube"]["accounts"].split(",")
        except KeyError:
            raise KeyError("Please add your youtube accounts to the secrets.toml file")

        if not accounts:
            raise ValueError("No youtube accounts found in the secrets.toml file")

        api_service_name = "youtube"
        api_version = "v3"

        # API client
        youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=accounts[0])

        request = youtube.search().list(part="id,snippet", q=search_query, type="video", maxResults=count)
        response = request.execute()["items"]

        yield format_response(response)
