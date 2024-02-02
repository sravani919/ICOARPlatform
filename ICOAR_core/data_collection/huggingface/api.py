from ..utils import BaseDataCollector


class Collector(BaseDataCollector):
    def __init__(self):
        super().__init__()

    @property
    def query_options(self):
        return NotImplementedError

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
