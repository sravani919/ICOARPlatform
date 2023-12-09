import string

import torch
from tqdm import tqdm
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


def predictCovidModel(text, model, tokenizer):
    tokens = tokenizer.encode_plus(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = outputs.logits.argmax(dim=1)
    predicted_labels = "Non-offensive" if predictions[0] == 0 else "Offensive"
    return predicted_labels


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


def validate(df, MODEL, tokenizer=None, label="label"):
    """
    Uses the given model to predict the labels of the given dataframe
    :param df: A pandas dataframe containing the data to validate
    :param MODEL: A string representing the model to use
    :param tokenizer: A string representing the tokenizer to use, default to none for autotokenizer
    :return: The dataframe with the predicted labels
    """
    if df is None:
        return
    if MODEL == "covid-twitter-bert":
        model_config = BertConfig.from_json_file("model/config.json")
        model_state_dict = torch.load("../../model/pytorch_model.bin", map_location=torch.device("cpu"))
        model = BertForSequenceClassification(config=model_config)
        model.load_state_dict(model_state_dict)
        tokenizer = BertTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
    else:
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL)
            except Exception as e:
                print(e)
                return

        if MODEL == "predictCovid":
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL, id2label={0: "Negative", 1: "Neutral", 2: "Positive"}
            )
        elif MODEL == "dslim/bert-base-NER":
            model = AutoModelForTokenClassification.from_pretrained(MODEL)
            nlp = tpipeline("ner", model=model, tokenizer=tokenizer)
        elif MODEL == "QCRI/bert-base-multilingual-cased-pos-english":
            model = AutoModelForTokenClassification.from_pretrained(MODEL)
            pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
        elif MODEL == "arpanghoshal/EmoRoBERTa":
            model = AutoModelForSequenceClassification.from_pretrained(MODEL, from_tf=True)

        else:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(MODEL)
            except Exception as e:
                print(e)
                return

    total_rows = len(df)
    progress_bar = tqdm(total=total_rows, desc="Processing rows", unit="row")

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
            continue
        if MODEL == "covid-twitter-bert":
            predicted_value = predictCovidModel(row["text"], model, tokenizer)
        else:
            predicted_value = predict(row["text"], model, tokenizer)
        df.loc[index, label] = predicted_value
        progress_bar.update(1)
    progress_bar.close()
    return df


class Validation:
    def __init__(self):
        pass

    def label(self, df, model, tokenizer=None, label="label"):
        return validate(df, model, tokenizer=tokenizer, label=label)
