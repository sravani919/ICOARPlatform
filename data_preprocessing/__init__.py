import os
import pandas as pd
import requests
from better_profanity import profanity

# --- Try to import tweet-preprocessor, but don't crash if missing ---
try:
    import preprocessor as p  # type: ignore
except ImportError:
    p = None

# --- Try to import fasttext, but don't crash if missing ---
try:
    import fasttext  # type: ignore
except ImportError:
    fasttext = None

# --- Try to load spaCy model, but don't crash if missing ---
try:
    import spacy  # type: ignore

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # If model not available, disable spaCy-based features
        nlp = None
except ImportError:
    spacy = None  # type: ignore
    nlp = None

# from profanity_filter import ProfanityFilter


def download_fasttext_model():
    """Download the FastText language identification model if not present."""
    response = requests.get(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    )
    # Need to create the models directory if it doesn't exist
    if not os.path.exists("./models"):
        os.makedirs("./models")

    with open("./models/lid.176.bin", "wb") as f:
        f.write(response.content)


# --- Try to load fasttext model if fasttext is available ---
fmodel = None
if fasttext is not None:
    try:
        fmodel = fasttext.load_model("./models/lid.176.bin")
    except Exception:
        print("Downloading fasttext language detection model...")
        download_fasttext_model()
        fmodel = fasttext.load_model("./models/lid.176.bin")

PUNCTUATION = [
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
    "-",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "'",
    '"',
    "\n",
    "\t",
    "\r",
    "\\",
    "/",
]

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
    "Remove Profanity",
]


def preprocess(filename, given_options):
    df = pd.read_csv(filename)

    # -------------------------------------------------
    # Ensure a 'text' column exists (CRITICAL FIX)
    # -------------------------------------------------
    if "text" not in df.columns:
        if "post_text" in df.columns:
            df["text"] = df["post_text"]
        elif "title" in df.columns:
            df["text"] = df["title"]
        else:
            raise ValueError(
                f"No usable text column found. Columns available: {df.columns.tolist()}"
            )


    # Shift rows to fill gaps in the index from the removed rows
    df.reset_index(drop=True, inplace=True)

    return True, df


def none_avoidance(func):
    def wrapper(row):
        if "text" not in row or isinstance(row["text"], float):
            return ""
        return func(row)
    return wrapper



def filter_non_english(row):
    # If fasttext model isn't available, don't filter anything out.
    if fmodel is None:
        return True

    if isinstance(row["text"], float):  # A float means the text is empty
        return False

    lang = (
        fmodel.predict(row["text"].replace("\n", ""))[0][0].replace("__label__", "")
    )
    return lang == "en"


def filter_no_text(row):
    return row["text"] != "" and isinstance(row["text"], str)


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
    # If spaCy / model not available, just return original text
    if nlp is None:
        return row["text"]

    doc = nlp(row["text"])
    return " ".join([token.text for token in doc if not token.is_stop])


@none_avoidance
def lemmatize_words(row):
    # If spaCy / model not available, just return original text
    if nlp is None:
        return row["text"]

    doc = nlp(row["text"])
    return " ".join([token.lemma_ for token in doc])


@none_avoidance
def remove_extra_spaces(row):
    return " ".join(row["text"].split())


@none_avoidance
def remove_url_hashtags_mentions(row):
    # If preprocessor isn't available, just return original text
    if p is None:
        return row["text"]
    return p.clean(row["text"])


@none_avoidance
def remove_profanity(row):
    # pf = ProfanityFilter()
    return profanity.censor(row["text"])
