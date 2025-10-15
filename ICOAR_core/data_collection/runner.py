# ICOAR_core/data_collection/runner.py

from ICOAR_core import data_collection
from ICOAR_core.data_collection.utils import ProgressUpdate, save_data

def collect_data(platform_name: str, method_name: str, query_values: dict, save_name: str = None):
    """
    Collects data using the collector logic from Streamlit backend
    """
    try:
        # Load the platform module like twitter, reddit, etc.
        platform_module = getattr(data_collection, platform_name.lower())
        
        # Get the collector class from the platform
        collector = platform_module.collection_methods[method_name].Collector()

        # Start collecting data using the generator
        results = []
        gen = collector.collect_generator(**query_values)
        for data in gen:
            if not isinstance(data, ProgressUpdate):
                results = data
                break

        if not results:
            return None, None

        if not save_name:
            save_name = f"{platform_name}_{query_values.get('keywords', 'data')}"

        save_data(results, save_name)
        return save_name + ".csv", results

    except Exception as e:
        raise Exception(f"Error during collection: {e}")

