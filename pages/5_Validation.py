import glob
import os
import nlu
import pandas as pd
import streamlit as st
import torch
from huggingface_hub import HfApi
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
)
# import spark_models
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

title = "Validation"
st.set_page_config(page_title=title)

placeholder = st.empty()
container = st.container()

st.sidebar.header(title)

if "output" not in st.session_state:
    st.session_state.output = pd.DataFrame()
if "model_list" not in st.session_state:
    st.session_state.model_list = []
if "predict" not in st.session_state:
    st.session_state.predict = False

FILE = st.sidebar.selectbox("Select a file", [file for file in glob.glob("./data/*.csv")])


MODELS = {
    "cardiffnlp/twitter-roberta-base-sentiment": {
        "tokenizer": AutoTokenizer,
        "model": "cardiffnlp/twitter-roberta-base-sentiment",
    },
    "Toxic hugging face": {"tokenizer": AutoTokenizer, "model": "s-nlp/roberta_toxicity_classifier"},
    "Seethal/sentiment_analysis_generic_dataset": {
        "tokenizer": AutoTokenizer,
        "model": AutoModelForSequenceClassification,
    },
    "Cyberbullying": {"NLU": "", "model": "en.classify.cyberbullying"},
    "Fakenews": {"NLU": "", "model": "en.classify.fakenews"},
    "Toxic": {"NLU": "", "model": "en.classify.toxic"},
}

MODEL = st.sidebar.radio("Select a model", list(MODELS.keys()))


def load_model(model_info):
    if "tokenizer" in model_info:
        pass

    else:
        loaded_model = nlu.load(model_info["model"])
        return loaded_model


def predict(text, model_info, model):
    if "tokenizer" in model_info:
        # tokenizer, model = model_info["tokenizer"], model_info["model"]
        tokenizer = AutoTokenizer.from_pretrained(model_info["model"])
        model = AutoModelForSequenceClassification.from_pretrained(model_info["model"])
        config = AutoConfig.from_pretrained(model_info["model"])

        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        output = outputs.logits.argmax().item()
        label = config.id2label[output]

        return label
    else:
        result = model.predict(text)
        result = result.drop("sentence_embedding_use", axis=1)
        # result = result.drop("sentence", axis=1)
        # column_names = result.columns.tolist()

    return result

def predictCovidModel(text, model, tokenizer):
    tokens = tokenizer.encode_plus(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = outputs.logits.argmax(dim=1)
    # label_names = ["non-offensive", "offensive"]
    predicted_labels = "Non-offensive" if predictions[0] == 0 else "Offensive"
    return predicted_labels


if st.sidebar.button("Predict"):
    df = pd.read_csv(FILE)

    model_info = MODELS[MODEL]

    model = load_model(model_info)

    for index, row in df.iterrows():
        if "tokenizer" in model_info:
            df.loc[index, "sentiment"] = predict(row["text"], model_info, [])

        else:
            result = predict([row["text"]], [], model)

            df.loc[index, result.columns] = result.values[0]

        with placeholder.container():
            st.write(df)
