import os

import pandas as pd

from ..utils import BaseDataCollector


class Collector(BaseDataCollector):
    def __init__(self):
        super().__init__()

    def query_options(self):
        return ["kaggle_json_file", "kaggle_dataset"]

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

        kaggle_json_info = kwargs.get("kaggle_json_file")
        kaggle_dataset = kwargs.get("kaggle_dataset")

        # Overwrites the kaggle.json file in the .kaggle directory in the user's home directory\
        home_dir = os.path.expanduser("~")

        # Create the .kaggle directory if it does not exist
        os.makedirs(f"{home_dir}/.kaggle", exist_ok=True)
        with open(f"{home_dir}/.kaggle/kaggle.json", "w") as f:
            f.write(kaggle_json_info)

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
        # os.system(f"rm -rf {save_dir}")

        yield out
