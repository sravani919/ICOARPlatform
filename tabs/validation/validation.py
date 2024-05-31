import os
import string
from io import StringIO

import pandas as pd
import streamlit as st
import torch
import transformers
from huggingface_hub import HfApi
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    TokenClassificationPipeline,
)
from transformers import pipeline as tpipeline

import tabs.Data_Collection.data_upload as data_upload


def initialize_state():
    if "output" not in st.session_state:
        st.session_state.output = pd.DataFrame()
    if "model_list" not in st.session_state:
        st.session_state.model_list = []
    if "predict" not in st.session_state:
        st.session_state.predict = False
    if "freq" not in st.session_state:
        st.session_state.freq = []
    if "disabled" not in st.session_state:
        st.session_state.disabled = True
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = "Recommended Models"
    if "current_model" not in st.session_state:
        st.session_state.current_model = None


def fetch_models_from_hf(search_text):
    # Or configure a HfApi client
    hf_api = HfApi(
        endpoint="https://huggingface.co",  # Can be a Private Hub endpoint.
        token=st.secrets.api_token.hf,  # Token is not persisted on the machine.
    )

    print("Fetching model list from hugging face...")
    models = list(hf_api.list_models(filter="text-classification", search=search_text))
    # sort models by downloads to get the user higher quality models.
    models.sort(key=lambda model: model.downloads, reverse=True)

    model_list = []

    for model in models:
        model_list.append(model.modelId)

    return model_list


def predictCovidModel(text, model, tokenizer):
    tokens = tokenizer.encode_plus(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = outputs.logits.argmax(dim=1)
    # label_names = ["non-offensive", "offensive"]
    predicted_labels = "Non-offensive" if predictions[0] == 0 else "Offensive"
    return predicted_labels


def save_file(df, filename):
    username = st.session_state["username"]
    if not os.path.exists("predicted"):
        os.makedirs("predicted")
    if not os.path.exists(f"""predicted/{username}"""):
        os.makedirs(f"""predicted/{username}""")
    file_path = f"predicted/{username}/{filename}.csv"
    df.to_csv(file_path, index=False)

    return file_path


def get_csv_string(df):
    """Converts the DataFrame to a CSV string, which can be used for downloading."""
    csv = StringIO()
    df.to_csv(csv, index=False)
    return csv.getvalue()


def predict(text, model, tokenizer):
    if (
        type(model) == transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification
        or type(model) == transformers.models.distilbert.modeling_distilbert.DistilBertForSequenceClassification
    ):
        inputs = tokenizer(text, return_tensors="pt", max_length=512, padding=True, truncation=True)
    else:
        inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    output = outputs.logits.argmax().item()

    config = model.config
    if hasattr(config, "id2label"):
        label = config.id2label[output]
    else:
        label = output

    return label


def validation():
    initialize_state()
    # FILE = st.selectbox("Select a file", [file for file in glob.glob("./data/*.csv")])
    email = st.session_state["username"]
    FILE = data_upload.data_upload_element(email, get_filepath_instead=True)
    freq = [0] * 28
    MODEL = ""
    MODELS = {}
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

    # selected_option = option_menu(
    #     None,
    #     text_menu_options,
    #     icons=["house", "cloud-upload", "list-task", "gear"],
    #     menu_icon="cast",
    #     default_index=0,
    #     orientation="horizontal",
    #     styles={"nav-link-selected": {"background-color": "red"}},
    # )
    cols = st.columns(2)

    with cols[0]:
        st.session_state.selected_option = st.radio(
            "Select classification model type", ["Recommended Models", "Search on Huggingface"]
        )
    with cols[1]:
        if st.session_state.selected_option == "Recommended Models":
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
            if st.session_state.current_model != selected_model_name:
                st.session_state.predict = False
                st.session_state.current_model = selected_model_name

            MODELS = MODELS[selected_model_name]
            MODEL = MODELS["model"]
            st.session_state.disabled = False

        elif st.session_state.selected_option == "Search on Huggingface":
            multi = """:bulb: **Steps** -
            \n 1. Input your Huggingface API token in the secrets.toml file. If you don't have
            one, you can get one [here](https://huggingface.co/settings/tokens)
            \n2. Input the name of the model you want to use. You can also just input a related keyword if you
            are not sure of which one to use.
            \n3. After that, click on the select/search button and verify that we are using the right model. If you
            searched by keyword, select the model you want to use from the list that appears. If you want to see more
             information about your selected model, there will be an expander below with a link to the model's page.
             These pages usually have a demo on the right side of the page, so that you can test the model before using
             it."""
            st.markdown(multi)
            choice = st.radio("Select an option:", ["Use a specific model", "Search by keyword"])
            if choice == "Use a specific model":
                model_name = st.text_input("Enter the name of the huggingface model (not the URL!):")
                model_button = st.button("Select")
                if model_button:
                    st.session_state.model_list = [model_name]
                    st.session_state.disabled = False
                    if st.session_state.current_model != model_name:
                        st.session_state.predict = False
                        st.session_state.current_model = model_name
                if st.session_state.model_list:
                    MODEL = st.radio("Model: ", [st.session_state.model_list[0]])
                    st.markdown("-------------------")
                    st.write(f"Verify we have the right model: [{MODEL}](https://huggingface.co/{MODEL})")

            if choice == "Search by keyword":
                search_text = st.text_input("Enter model name")
                search_button = st.button("Search")
                if search_button:
                    st.session_state.model_list = fetch_models_from_hf(search_text)
                    search_button = False
                    st.session_state.disabled = False
                if st.session_state.model_list:
                    MODEL = st.radio(
                        "Top Three Models:",
                        st.session_state.model_list[:3],
                    )
                    # display the rest of the models if the user wants to see more
                    if st.checkbox("Show more"):
                        MODEL = st.radio(
                            "All Results",
                            st.session_state.model_list[3:],
                        )
                    if st.session_state.current_model != MODEL:
                        st.session_state.predict = False
                        st.session_state.current_model = MODEL
                if MODEL:
                    with st.expander("Model Details"):
                        model_url = f"https://huggingface.co/{MODEL}"
                        st.write(f"Model URL: [{MODEL}]({model_url})")

    # prevents users from initially clicking predict button without choosing a model
    if st.session_state.disabled:
        st.warning("Please select a model to proceed")
        return
    predict_button = st.button("Predict")
    st.markdown("-------------------")
    if predict_button:
        placeholder = st.empty()
        st.session_state.predict = True
        df = pd.read_csv(FILE)
        total_rows = df.shape[0]

        if MODEL == "covid-twitter-bert":
            model_config = BertConfig.from_json_file("model/config.json")
            model_state_dict = torch.load("model/pytorch_model.bin", map_location=torch.device("cpu"))
            model = BertForSequenceClassification(config=model_config)
            model.load_state_dict(model_state_dict, strict=False)
            tokenizer = BertTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
        else:
            with st.spinner("Downloading necessary models. It may take few minutes. Please wait..."):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(MODEL)
                except Exception as e:
                    st.error(e)
                    return

                if "id2label" in MODELS:
                    model = AutoModelForSequenceClassification.from_pretrained(MODEL, id2label=MODELS["id2label"])
                    tokenizer = MODELS["tokenizer"].from_pretrained(MODEL)
                elif MODEL == "dslim/bert-base-NER":
                    model = AutoModelForTokenClassification.from_pretrained(MODEL)
                    nlp = tpipeline("ner", model=model, tokenizer=tokenizer)
                elif MODEL == "QCRI/bert-base-multilingual-cased-pos-english":
                    model = AutoModelForTokenClassification.from_pretrained(MODEL)
                    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
                elif MODEL == "arpanghoshal/EmoRoBERTa":
                    model = AutoModelForSequenceClassification.from_pretrained(MODEL, from_tf=True)

                else:
                    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

        progress_bar = st.empty()

        for index, row in df.iterrows():
            if pd.isnull(row["text"]):
                continue
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

    if st.session_state.predict:
        if MODEL == "arpanghoshal/EmoRoBERTa":
            with st.container():
                filename = st.text_input("Enter file name to save predicted data")
                save = st.button("Save File")
                if save:
                    file_path = save_file(st.session_state.output, filename)
                    st.session_state.predict = False
                    st.success("Saved to '" + file_path + "'")

                csv_data = get_csv_string(st.session_state.output)
                st.download_button(
                    label="Download",
                    data=csv_data,
                    file_name=f"{filename}.csv",
                    mime="text/csv",
                    help="Click to download the CSV file with predicted data.",
                )

            # elif selected_tab == "Emotional Analysis":
            #     emotional_analysis(st.session_state.output)
        else:
            filename = st.text_input("Enter file name  to save predicted data")
            save = st.button("Save File")
            if save:
                file_path = save_file(st.session_state.output, filename)
                st.session_state.predict = False
                st.success("Saved to '" + file_path + "'")

            csv_data = get_csv_string(st.session_state.output)
            st.download_button(
                label="Download",
                data=csv_data,
                file_name=f"{filename}.csv",
                mime="text/csv",
                help="Click to download the CSV file with predicted data.",
            )
