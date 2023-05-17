import fasttext
import pandas as pd
import preprocessor as p

fmodel = fasttext.load_model("./models/lid.176.bin")


def preprocess(filename, english_only, tweet_clean,drop_empty):
    df = pd.read_csv(filename)
    if english_only:
        m = df.apply(filter_non_english, axis=1)
        df = df[m]
    if tweet_clean:
        df["text"] = df["text"].apply(p.clean)
    if drop_empty:
        df.dropna(how='all', inplace=True)
        df["text"] = df["text"].astype(str).str.lower()
 

        
    df.to_csv(filename, index=False)
    return True


def filter_non_english(row):
    lang = fmodel.predict(row["text"].replace("\n", ""))[0][0].replace("__label__", "")
    return lang == "en"
