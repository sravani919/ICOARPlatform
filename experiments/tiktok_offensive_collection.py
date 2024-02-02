import os
import time

import pandas as pd
import toml

from ICOAR_core.data_collection.tiktok.api import TikTokApi

if __name__ == "__main__":
    # loading all of the or keywords from offensive_keywords.csv
    try:
        offensive_keywords = pd.read_csv("keyword/offensive_keywords.csv")
    except FileNotFoundError as e:
        print("offensive_keywords.csv file can be added with the keywords you want to search for")
        raise e

    offensive_keywords = offensive_keywords.iloc[:, 0]
    offensive_keywords = offensive_keywords.tolist()

    print(offensive_keywords)

    # loading the secrets.toml file to get the client key and secret
    # This can be hard-coded instead
    secrets = toml.load("../.streamlit/secrets.toml")
    tiktok = TikTokApi(secrets["tiktok"]["client_key"], secrets["tiktok"]["client_secret"])

    results, si = tiktok.video_request(
        max_count=100000,
        keywords=None,
        keywordsOR=offensive_keywords,
        start_date="20210601",
        end_date="20210630",
        locations=None,
        hashtags=None,
    )
    df = pd.DataFrame(results)
    # Saving to csv with the time stamp included in mm-dd-yyyy-hh-mm-ss format
    if not os.path.exists("data"):
        os.mkdir("data")
    df.to_csv("data/tiktok_" + time.strftime("%m-%d-%Y-%H-%M-%S") + ".csv", index=False)
