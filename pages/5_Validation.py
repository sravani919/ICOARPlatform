import glob

import pandas as pd
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

title = "Validation"

st.set_page_config(page_title=title)

placeholder = st.empty()
container = st.container()

st.sidebar.header(title)

if "output" not in st.session_state:
    st.session_state.output = None

FILE = st.sidebar.selectbox("Select a file", [file for file in glob.glob("./data/*.csv")])

MODEL = st.sidebar.radio(
    "Select a model",
    [
        "cardiffnlp/twitter-roberta-base-sentiment",
        "finiteautomata/bertweet-base-sentiment-analysis",
        "Seethal/sentiment_analysis_generic_dataset",
    ],
)


def predict(text):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    output = outputs.logits.argmax().item()
    # Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
    label = "Negative" if output == 0 else "Neutral" if output == 1 else "Positive"
    return label


if st.sidebar.button("Predict"):
    df = pd.read_csv(FILE)
    for index, row in df.iterrows():
        df.loc[index, "sentiment"] = predict(row["text"])
        with placeholder.container():
            st.write(df)
