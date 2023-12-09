import pandas as pd

from ICOAR_core import validation

# collector = data_collection.facebook.scraper.Collector()
# results = collector.collect(["party"], 5)
# print(results)

# path ICOAR/data/trump copy.csv
# load this as a dataframe
df = pd.read_csv("data/trump copy.csv")
# validate the dataframe
validator = validation.validation_package.Validation()
results = validator.label(df, "cardiffnlp/twitter-roberta-base-sentiment", None, "sentiment")
print(results.to_string())
