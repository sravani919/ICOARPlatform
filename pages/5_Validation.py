import glob

import pandas as pd
import streamlit as st
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

title = "Validation"
st.set_page_config(page_title=title)

placeholder = st.empty()
container = st.container()

st.sidebar.header(title)

if "output" not in st.session_state:
    st.session_state.output = None
if "model_list" not in st.session_state:
    st.session_state.model_list = []

FILE = st.sidebar.selectbox("Select a file", [file for file in glob.glob("./data/*.csv")])

search_text = st.sidebar.text_input("Enter model name")
search_button = st.sidebar.button("Search")


def fetch_models_from_hf():
    # Or configure a HfApi client
    hf_api = HfApi(
        endpoint="https://huggingface.co",  # Can be a Private Hub endpoint.
        token="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",  # Token is not persisted on the machine.
    )

    print("Fetching model list from hugging face...")
    models = hf_api.list_models(filter="text-classification", search=search_text)
    model_list = []
    for model in models:
        model_list.append(model.modelId)

    return model_list


if search_button:
    st.session_state.model_list = fetch_models_from_hf()
    search_button = False

MODEL = st.sidebar.radio(
    "Select a model",
    st.session_state.model_list,
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
