import pandas as pd
from datasets import load_dataset

from ..utils import BaseDataCollector


class Collector(BaseDataCollector):
    def __init__(self):
        super().__init__()

    def query_options(self):
        return ["huggingface_dataset"]

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

        hf_dataset = hf_dataset_url.split("huggingface.co/datasets/")[1]

        dataset = load_dataset(hf_dataset)

        df = pd.DataFrame(dataset["train"])

        # Convert df to list of dictionaries
        out = df.to_dict(orient="records")

        yield out
