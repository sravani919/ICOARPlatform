import pandas as pd
from datasets import load_dataset

dataset = load_dataset("deberain/ChatGPT-Tweets")

# Save as csv
df = pd.DataFrame(dataset["train"])
df.to_csv("imdb.csv", index=False)
