import json
import os
import shutil

import pandas as pd
import toml

from ..utils import BaseDataCollector


def cant_find_keys():
    return """
Could not find Kaggle API credentials. Please add them to your secrets.toml file.
Visit https://www.kaggle.com/settings/account and click "Create New Token" to get a kaggle.json file which
contains your username and key.
Example Kaggle secrets.toml:

[kaggle]
username = "your_kaggle_username"
key = "your_kaggle_key"
"""


class Collector(BaseDataCollector):
    def __init__(self):
        super().__init__()

    def query_options(self):
        return ["kaggle_dataset", "delete_temp_data"]

    def auth(self) -> list[str]:
        return ["kaggle.username", "kaggle.key"]

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

        kaggle_dataset_url = kwargs.get("kaggle_dataset")
        delete_temp_data = kwargs.get("delete_temp_data")

        kaggle_dataset = kaggle_dataset_url.split("kaggle.com/datasets/")[1]

        try:
            # Loading secrets.toml to get Kaggle API credentials
            secrets = toml.load(".streamlit/secrets.toml")
            username = secrets["kaggle"]["username"]
            key = secrets["kaggle"]["key"]
        except Exception as e:
            raise ValueError(cant_find_keys() + f"\n{e}")

        # Overwrites the kaggle.json file in the .kaggle directory in the user's home directory
        home_dir = os.path.expanduser("~")

        # Create the .kaggle directory if it does not exist
        os.makedirs(f"{home_dir}/.kaggle", exist_ok=True)
        with open(f"{home_dir}/.kaggle/kaggle.json", "w") as f:
            json_string = json.dumps({"username": username, "key": key})
            f.write(json_string)

        # Import package here now that the kaggle.json file has been written
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kaggle.api_client import ApiClient

        # Download the dataset to a directory called kaggle_data/owner/dataset
        save_dir = f"data/kaggle_temp_data/{kaggle_dataset.split('/')[0]}/{kaggle_dataset.split('/')[1]}"
        os.makedirs(save_dir, exist_ok=True)
        api = KaggleApi(ApiClient())
        api.authenticate()

        api.dataset_download_files(kaggle_dataset, path=save_dir, force=True, unzip=True)

        # Open the directory and read the first file
        files_downloaded = os.listdir(save_dir)

        # Convert the first file to a list of dictionaries
        df = pd.read_csv(f"{save_dir}/{files_downloaded[0]}")
        out = df.to_dict(orient="records")

        # Delete the kaggle_temp_data
        if delete_temp_data:
            shutil.rmtree(save_dir)

        yield out
