import os
import socket
import urllib.error
from urllib.request import HTTPCookieProcessor, build_opener

import pandas as pd
import requests
from PIL import Image

IMAGE_DOWNLOAD_MAX_ATTEMPTS = 3

mike = {
    "gmail_username": "farmermike876@gmail.com",
    "gmail_password": "FarmerMike876!",
    "facebook_password": "FarmerMike876!",
    "tiktok_username": "farmermike876",
    "birthday": "September 3, 1980",
    "gender": "male",
}


def download_image(url):
    """
    Downloads an image from the given url
    :param url: The direct url to the image
    :return: PIL Image object or None if the image couldn't be downloaded
    """
    if url is None:
        return

    opener = build_opener(HTTPCookieProcessor())

    attempt = 0
    while attempt < IMAGE_DOWNLOAD_MAX_ATTEMPTS:
        try:
            response = opener.open(url, timeout=2)
            image = Image.open(response).convert("RGB")
            return image
        except urllib.error.URLError:
            print(f"URLError opening {url}")
        except socket.timeout:
            print(f"Socket timeout opening {url}")
        attempt += 1
        print(f"Retrying... ({attempt}/{IMAGE_DOWNLOAD_MAX_ATTEMPTS})")

    return


def save_data(posts, filename, folder_path="data"):
    import streamlit as st

    df = pd.DataFrame(posts)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    username = st.session_state["username"]
    file_path = f"{folder_path}/{username}/{filename}.csv"
    df.to_csv(file_path, index=False)
    return file_path


def download_images(posts, filename, i):
    import streamlit as st

    """
    Downloads the images from the given posts to the given file path
    :param posts: The list of post dictionaries
    :param filename: The filename being used to save the posts
    :param i: The index of the post to download the images from
    """
    post = posts[i]
    images_path = ""

    for i, url in enumerate(post["image_urls"]):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                file_name = f"{post['id']}_{i}.jpg"
                username = st.session_state["username"]
                images_path = f"data/{username}/{filename}_images"
                if not os.path.exists(images_path):
                    os.makedirs(images_path)
                file_path = os.path.join(images_path, file_name)
                with open(file_path, "wb") as file:
                    file.write(response.content)
        except requests.exceptions.RequestException:
            continue

    return images_path


class BaseDataCollector:
    """
    This class is Abstract and should be inherited by all data collectors.
    Some example data collectors are the Twitter Scraper, Facebook Scraper, YouTube Comment Scraper, TikTok API, etc.
    """

    @property
    def query_options(self):
        """
        Should return a list of query options such as ['count', 'keywords', 'start_date', 'end_date']
        """
        raise NotImplementedError

    def collect(self, *args, **kwargs):
        """
        Should take in the query options and return a list of dictionaries with each dictionary
        being a singe result.
        """
        raise NotImplementedError

    def collect_generator(self, *args, **kwargs):
        """
        Should take in the query options
        Will yield progress updates
        The final yield should be a list of dictionaries with each dictionary being a singe result.

        The progress yields will be of type ProgressUpdate which contains a float between 0 and 1 representing the
        progress and a string describing the progress

        @yields ProgressUpdate IFF not done
        @yields list[dict] IFF done
        """
        raise NotImplementedError


class ProgressUpdate:
    """
    A class representing a progress update
    Used to tell if a yield is a progress update or the final data

    These should be yielded by collect_generator when it hasn't finished yet
    These should be used to update a progress bar
    """

    def __init__(self, progress: float, text: str):
        """
        :param progress: A float between 0 and 1 representing the current progress
        :param text: A string describing the current progress
        """
        self.progress = progress
        self.text = text
