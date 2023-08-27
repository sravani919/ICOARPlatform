import glob
import os
import string

import pandas as pd
import streamlit as st
import torch
from huggingface_hub import HfApi
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    TokenClassificationPipeline,
    pipeline,
)

from emotional_analysis import emotional_analysis

title = "Validation"
st.set_page_config(page_title=title)

container = st.container()

st.subheader(title)
FILE = st.selectbox("Select a file", [file for file in glob.glob("./data/*.csv")])

if "output" not in st.session_state:
    st.session_state.output = pd.DataFrame()
if "model_list" not in st.session_state:
    st.session_state.model_list = []
if "predict" not in st.session_state:
    st.session_state.predict = False
if "freq" not in st.session_state:
    st.session_state.freq = []
freq = [0] * 28
categories = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

st.session_state.horizontal = True
model_type = st.radio(
    "Select a model type",
    ["Recommended Models", "Search on huggingface"],
    horizontal=st.session_state.horizontal,
    help="You will need to input a huggingface API token to search via huggingface, or "
    "you can select from the recommended models that have already been set up",
)


def fetch_models_from_hf():
    # Or configure a HfApi client
    hf_api = HfApi(
        endpoint="https://huggingface.co",  # Can be a Private Hub endpoint.
        token=st.secrets.api_token.hf,  # Token is not persisted on the machine.
    )

    print("Fetching model list from hugging face...")
    models = hf_api.list_models(filter="text-classification", search=search_text)
    model_list = []
    for model in models:
        model_list.append(model.modelId)

    return model_list


if model_type == "Search on huggingface":
    search_text = st.text_input("Enter model name")
    search_button = st.button("Search")

    if search_button:
        st.session_state.model_list = fetch_models_from_hf()
        search_button = False

    MODEL = st.radio(
        "Select a model",
        st.session_state.model_list,
    )
else:
    MODELS = {
        "Covid offensive tweets Detection": {"model": "covid-twitter-bert"},
        "Sentiment Analysis": {
            "tokenizer": AutoTokenizer,
            "model": "cardiffnlp/twitter-roberta-base-sentiment",
            "id2label": {0: "Negative", 1: "Neutral", 2: "Positive"},
        },
        "Toxic Content Detection": {
            "tokenizer": AutoTokenizer,
            "model": "s-nlp/roberta_toxicity_classifier",
        },
        "Hate Speech Detection": {
            "tokenizer": AutoTokenizer,
            "model": "cardiffnlp/twitter-roberta-base-hate-latest",
        },
        "Cyberbully Detection": {
            "tokenizer": AutoTokenizer,
            "model": "sreeniketh/cyberbullying_sentiment_dsce_2023",
        },
        "Named Entity Recognition": {
            "tokenizer": AutoTokenizer,
            "model": "dslim/bert-base-NER",
        },
        "Parts of Speech": {
            "tokenizer": AutoTokenizer,
            "model": "QCRI/bert-base-multilingual-cased-pos-english",
        },
        "Emotion Analysis": {
            "tokenizer": AutoTokenizer,
            "model": "arpanghoshal/EmoRoBERTa",
        },
    }

    selected_model_name = st.selectbox("Select a model", list(MODELS.keys()))

    MODELS = MODELS[selected_model_name]
    MODEL = MODELS["model"]


def save_file(df, filename):
    if not os.path.exists("predicted"):
        os.makedirs("predicted")
    file_path = f"predicted/{filename}.csv"
    df.to_csv(file_path, index=False)
    return file_path


def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    output = outputs.logits.argmax().item()

    config = model.config
    if hasattr(config, "id2label"):
        label = config.id2label[output]
    else:
        label = output

    return label


def predictCovidModel(text, model, tokenizer):
    tokens = tokenizer.encode_plus(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = outputs.logits.argmax(dim=1)
    # label_names = ["non-offensive", "offensive"]
    predicted_labels = "Non-offensive" if predictions[0] == 0 else "Offensive"
    return predicted_labels


if st.button("Predict"):
    placeholder = st.empty()
    st.session_state.predict = True
    df = pd.read_csv(FILE)
    total_rows = df.shape[0]

    if MODEL == "covid-twitter-bert":
        model_config = BertConfig.from_json_file("model/config.json")
        model_state_dict = torch.load("model/pytorch_model.bin", map_location=torch.device("cpu"))
        model = BertForSequenceClassification(config=model_config)
        model.load_state_dict(model_state_dict)
        tokenizer = BertTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
    else:
        with st.spinner("Downloading necessary models. It may take few minutes. Please wait..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL)

            if "id2label" in MODELS:
                model = AutoModelForSequenceClassification.from_pretrained(MODEL, id2label=MODELS["id2label"])
            elif MODEL == "dslim/bert-base-NER":
                model = AutoModelForTokenClassification.from_pretrained(MODEL)
                nlp = pipeline("ner", model=model, tokenizer=tokenizer)
            elif MODEL == "QCRI/bert-base-multilingual-cased-pos-english":
                model = AutoModelForTokenClassification.from_pretrained(MODEL)
                pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
            elif MODEL == "arpanghoshal/EmoRoBERTa":
                model = AutoModelForSequenceClassification.from_pretrained(MODEL, from_tf=True)

            else:
                model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    progress_bar = st.empty()

    # print("model - ", MODEL)
    for index, row in df.iterrows():
        if MODEL == "dslim/bert-base-NER" or MODEL == "QCRI/bert-base-multilingual-cased-pos-english":
            if MODEL == "dslim/bert-base-NER":
                output = nlp(row["text"])
            elif MODEL == "QCRI/bert-base-multilingual-cased-pos-english":
                output = pipeline(row["text"])
            predicted_entities = {}
            for entity_info in output:
                entity_group = entity_info["entity"]
                word = entity_info["word"]

                if entity_group not in predicted_entities:
                    predicted_entities[entity_group] = []

                predicted_entities[entity_group].append(word)

            # Remove punctuation words from the predicted_entities
            for punctuation in string.punctuation:
                if punctuation in predicted_entities:
                    del predicted_entities[punctuation]

            for entity_group, words_list in predicted_entities.items():
                if entity_group not in df.columns:
                    df[entity_group] = ""
                df.at[index, entity_group] = ", ".join(words_list)  # Convert the list to a string

            with placeholder.container():
                progress = (index + 1) / total_rows
                progress_bar.progress(progress, text=f"Predicting text: {progress * 100:.2f}% complete")
            continue
        if MODEL == "arpanghoshal/EmoRoBERTa":
            predicted_value = predict(row["text"], model, tokenizer)
            df.loc[index, "emotion"] = predicted_value
            emotion_index = categories.index(predicted_value)
            freq[emotion_index] += 1

            with placeholder.container():
                progress = (index + 1) / total_rows
                progress_bar.progress(progress, text=f"Predicting text: {progress * 100:.2f}% complete")
            continue
        if MODEL == "covid-twitter-bert":
            predicted_value = predictCovidModel(row["text"], model, tokenizer)
        else:
            predicted_value = predict(row["text"], model, tokenizer)
        df.loc[index, "sentiment"] = predicted_value
        with placeholder.container():
            st.dataframe(df[["text", "sentiment"]][max(0, index - 10) : max(10, index)])
            progress = (index + 1) / total_rows
            progress_bar.progress(progress, text=f"Predicting text: {progress * 100:.2f}% complete")

    with placeholder.container():
        st.dataframe(df)
    progress_bar.empty()

    st.session_state.output = df
    st.success("Prediction completed", icon="✅")

if st.session_state.predict and MODEL == "arpanghoshal/EmoRoBERTa":
    tab1, tab2 = st.tabs(["Save Data", "Emotional Analysis"])
    with tab1:
        filename = st.text_input("Enter file name  to save predicted data")
        save = st.button("Save File")
        if save:
            file_path = save_file(st.session_state.output, filename)
            st.session_state.predict = False
            st.success("Saved to '" + file_path + "'")
    with tab2:
        emotional_analysis(st.session_state.output)

elif st.session_state.predict:
    filename = st.text_input("Enter file name  to save predicted data")
    save = st.button("Save File")
    if save:
        file_path = save_file(st.session_state.output, filename)
        st.session_state.predict = False
        st.success("Saved to '" + file_path + "'")
