import fasttext
import pandas as pd
import preprocessor as p
import spacy
import requests
import os

def download_fasttext_model():
    response = requests.get("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
    # Need to create the models directory if it doesn't exist
    if not os.path.exists("./models"):
        os.makedirs("./models")

    with open("./models/lid.176.bin", "wb") as f:
        f.write(response.content)


try:
    fmodel = fasttext.load_model("./models/lid.176.bin")
except ValueError:
    print("Downloading fasttext language detection model...")
    download_fasttext_model()
    fmodel = fasttext.load_model("./models/lid.176.bin")

nlp = spacy.load("en_core_web_sm")

PUNCTUATION = [".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "'", '"', "\n", "\t", "\r", "\\", "/"]

options = [
    "Remove non-English tweets",
    "Remove URLs, Hashtags, Mentions, Emojis",
    "Remove special characters",
    "Lemmatize",
    "Lowercase all words",
    "Remove stop words",
    "Remove empty texts",
    "Remove punctuation",
    "Remove extra spaces",
]


def preprocess(filename, given_options):
    df = pd.read_csv(filename)

    if given_options[0]:
        m = df.apply(filter_non_english, axis=1)
        df = df[m]
    if given_options[1]:
        # It's important that this one goes first, so the other options don't mess with the text in
        # a way that makes it harder to detect the URLs, Hashtags, and Mentions
        df["text"] = df.apply(remove_url_hashtags_mentions, axis=1)
    if given_options[2]:
        df["text"] = df.apply(strip_special_characters, axis=1)
    if given_options[3]:
        df["text"] = df.apply(lemmatize_words, axis=1)
    if given_options[4]:
        df["text"] = df.apply(lowercase_words, axis=1)
    if given_options[5]:
        df["text"] = df.apply(strip_stop_words, axis=1)
    if given_options[6]:
        df = df[df.apply(filter_no_text, axis=1)]
    if given_options[7]:
        df["text"] = df.apply(strip_punctuation, axis=1)
    if given_options[8]:  # Removes extra spaces
        df["text"] = df.apply(remove_extra_spaces, axis=1)

    # Shift rows the fill gaps in the index from the removed rows
    df.reset_index(drop=True, inplace=True)

    return True, df


def none_avoidance(func):  # Avoids errors from dealing with None values when expecting strings
    def wrapper(row):
        if type(row["text"]) is float:
            return ""
        else:
            return func(row)

    return wrapper


def filter_non_english(row):
    if type(row["text"]) is float:  # A float means the text is empty
        return False

    lang = fmodel.predict(row["text"].replace("\n", ""))[0][0].replace("__label__", "")
    return lang == "en"


def filter_no_text(row):
    return row["text"] != "" and type(row["text"]) == str


@none_avoidance
def lowercase_words(row):
    return row["text"].lower()


@none_avoidance
def strip_punctuation(row):
    return "".join([char for char in row["text"] if char not in PUNCTUATION])


@none_avoidance
def strip_special_characters(row):
    return "".join([char for char in row["text"] if char.isalnum() or char == " "])


@none_avoidance
def strip_stop_words(row):
    doc = nlp(row["text"])
    return " ".join([token.text for token in doc if not token.is_stop])


@none_avoidance
def lemmatize_words(row):
    doc = nlp(row["text"])
    return " ".join([token.lemma_ for token in doc])


@none_avoidance
def remove_extra_spaces(row):
    return " ".join(row["text"].split())


@none_avoidance
def remove_url_hashtags_mentions(row):
    return p.clean(row["text"])
