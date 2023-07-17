import socket
import urllib.error
from urllib.request import HTTPCookieProcessor, build_opener

import pandas as pd
from PIL import Image

IMAGE_DOWNLOAD_MAX_ATTEMPTS = 3


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
    df = pd.DataFrame(posts)
    file_path = f"{folder_path}/{filename}.csv"
    df.to_csv(file_path, index=False)
    return file_path
