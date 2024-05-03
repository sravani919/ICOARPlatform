import pandas as pd
from datasets import load_dataset

from ..utils import BaseDataCollector, ProgressUpdate


class Collector(BaseDataCollector):
    def __init__(self):
        super().__init__()

    def query_options(self):
        return ["huggingface_dataset"]

    def auth(self) -> list[str]:
        return []

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

        hf_dataset_url = kwargs.get("huggingface_dataset")
        kwargs.get("delete_temp_data")

        # Remove the huggingface.co/datasets/ part of the URL
        hf_dataset = hf_dataset_url.split("huggingface.co/datasets/")[1]

        yield ProgressUpdate(0, "Requesting dataset from huggingface...")
        # Grabbing the dataset from huggingface
        dataset = load_dataset(hf_dataset)

        yield ProgressUpdate(0.5, "Dataset retrieved")

        df = pd.DataFrame(dataset["train"])

        # Convert df to list of dictionaries
        out = df.to_dict(orient="records")
        yield ProgressUpdate(1, "Data collected")

        yield out
