import os
import string
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import torch
except Exception:
    torch = None

from huggingface_hub import HfApi
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
)
from transformers import pipeline as tpipeline

import tabs.Data_Collection.data_upload as data_upload


# ==========================================================
# UI HELPERS (Text Analysis — same style pattern as DC/PP/Viz)
# ==========================================================
def _ta_hero_header(active_step: int = 1):
    steps = [
        (1, "Select data"),
        (2, "Choose model"),
        (3, "Predict"),
        (4, "Export"),
    ]

    chips_html = '<div class="ta-steps">' + "".join(
        [
            f'<div class="ta-chip {"active" if s == active_step else ""}">'
            f'<span class="ta-chip-num">{s}</span>'
            f'<span class="ta-chip-name">{name}</span>'
            f"</div>"
            for s, name in steps
        ]
    ) + "</div>"

    st.markdown(
        f"""
<div class="ta-hero-bleed">
  <div class="ta-hero-inner">
    <div class="ta-hero-title">Text Analysis</div>
    <div class="ta-hero-sub">Choose a dataset, select a model, run prediction, and export results.</div>
    {chips_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _ta_step_card(title: str):
    wrap = st.container()
    with wrap:
        st.markdown('<div class="ta-card-marker"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ta-step-title">{title}</div>', unsafe_allow_html=True)
    return wrap


# -----------------------------
# SESSION STATE
# -----------------------------
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

    if "ta_step" not in st.session_state:
        st.session_state.ta_step = 1

    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None


# -----------------------------
# HF SEARCH
# -----------------------------
def fetch_models_from_hf(search_text):
    hf_api = HfApi(
        endpoint="https://huggingface.co",
        token=st.secrets.api_token.hf,
    )
    models = list(hf_api.list_models(filter="text-classification", search=search_text))
    models.sort(key=lambda model: model.downloads, reverse=True)
    return [m.modelId for m in models]


# -----------------------------
# HELPERS
# -----------------------------
def save_file(df, filename):
    username = st.session_state["username"]
    if not os.path.exists("predicted"):
        os.makedirs("predicted")
    if not os.path.exists(f"predicted/{username}"):
        os.makedirs(f"predicted/{username}")

    file_path = f"predicted/{username}/{filename}.csv"
    df.to_csv(file_path, index=False)
    return file_path


def get_csv_string(df):
    csv = StringIO()
    df.to_csv(csv, index=False)
    return csv.getvalue()


def predict(text, model, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    outputs = model(**inputs)
    output = outputs.logits.argmax().item()

    config = model.config
    if hasattr(config, "id2label"):
        return config.id2label[output]
    return output


# -----------------------------
# SAFETY HELPERS
# -----------------------------
def _normalize_text_column(df: pd.DataFrame) -> pd.DataFrame:
    if "text" in df.columns:
        return df
    if "post_text" in df.columns:
        df["text"] = df["post_text"]
        return df
    if "title" in df.columns:
        df["text"] = df["title"]
        return df
    if "comments" in df.columns:
        df["text"] = df["comments"].astype(str)
        return df
    return df


def _safe_from_pretrained_sequence(MODEL: str, **kwargs):
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL, use_safetensors=True, **kwargs
        )
    except Exception as e:
        st.error(
            f"Model '{MODEL}' does not support safetensors.\n\n"
            "Unsafe .bin loading is blocked in this environment.\n"
            "Please choose a model that provides *.safetensors*.\n\n"
            "Tip: HuggingFace → Files → look for .safetensors"
        )
        raise RuntimeError(e)


def _safe_from_pretrained_token(MODEL: str, **kwargs):
    try:
        return AutoModelForTokenClassification.from_pretrained(
            MODEL, use_safetensors=True, **kwargs
        )
    except Exception as e:
        st.error(
            f"Token model '{MODEL}' does not support safetensors.\n\n"
            "Unsafe torch.load() is blocked in this environment.\n"
            "Please select a safetensors-enabled model."
        )
        raise RuntimeError(e)


# ==========================================================
# ✅ AI ASSISTANT ENTRYPOINT (outside validation())
# ==========================================================
RECOMMENDED_TASK_MODELS = {
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment",
    "toxicity": "unitary/toxic-bert",
    "hate": "cardiffnlp/twitter-roberta-base-hate-latest",
    "cyberbullying": "sreeniketh/cyberbullying_sentiment_dsce_2023",
    "emotion": "arpanghoshal/EmoRoBERTa",
}


def run_text_analysis_file(input_path: str, task: str = "toxicity"):
    task = (task or "toxicity").lower().strip()
    if task not in RECOMMENDED_TASK_MODELS:
        raise ValueError(f"Unknown task: {task}. Choose from {list(RECOMMENDED_TASK_MODELS.keys())}")

    p = Path(input_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    df = pd.read_csv(str(p))
    df = _normalize_text_column(df)
    if "text" not in df.columns:
        raise RuntimeError(f"No usable text column found. Columns: {df.columns.tolist()}")

    MODEL = RECOMMENDED_TASK_MODELS[task]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    if MODEL == "arpanghoshal/EmoRoBERTa":
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL, from_tf=True, use_safetensors=True
            )
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(MODEL, from_tf=True)
        out_col = "emotion"
    elif MODEL == "cardiffnlp/twitter-roberta-base-sentiment":
        model = _safe_from_pretrained_sequence(MODEL)
        model.config.id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        out_col = "sentiment"
    else:
        model = _safe_from_pretrained_sequence(MODEL)
        out_col = task

    if torch is not None:
        model.eval()

    texts = df["text"].fillna("").astype(str).tolist()
    if torch is not None:
        with torch.no_grad():
            preds = [predict(t, model, tokenizer) for t in texts]
    else:
        preds = [predict(t, model, tokenizer) for t in texts]

    df[out_col] = preds
    out_path = p.with_name(p.stem + f"__{task}.csv")
    df.to_csv(str(out_path), index=False)
    return str(out_path), len(df), [out_col]


# ==========================================================
# MAIN UI: validation()
# ==========================================================
def validation(email=None):
    initialize_state()

    step = int(st.session_state.get("ta_step", 1))
    _ta_hero_header(active_step=step)
    st.markdown('<div class="ta-wrap">', unsafe_allow_html=True)

    # =========================
    # STEP 1 — SELECT DATA
    # =========================
    if st.session_state.ta_step == 1:
        card1 = _ta_step_card("1) Select data")
        with card1:
            df_from_pp = st.session_state.get("processed_df", None)
            FILE = None
            df_loaded = None

            if df_from_pp is not None and isinstance(df_from_pp, pd.DataFrame) and len(df_from_pp) > 0:
                st.success("Using cleaned dataset from Pre-processing.")
                df_loaded = df_from_pp.copy()
                st.dataframe(df_loaded, use_container_width=True, height=320)
            else:
                email = st.session_state["username"]
                FILE = data_upload.data_upload_element(email, get_filepath_instead=True)
                if not FILE:
                    st.info("Select a dataset to continue.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                try:
                    df_loaded = pd.read_csv(FILE)
                    st.dataframe(df_loaded, use_container_width=True, height=320)
                except Exception as e:
                    st.error("Failed to read the selected dataset.")
                    st.exception(e)
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

            df_loaded = _normalize_text_column(df_loaded)

            spacer, next_col = st.columns([0.76, 0.24])
            with next_col:
                if st.button("Next →", key="ta_next_1", type="primary", use_container_width=True):
                    st.session_state._ta_df_loaded = df_loaded
                    st.session_state._ta_file_path = FILE or ""
                    st.session_state.output = pd.DataFrame()
                    st.session_state.ta_step = 2
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        return

    # =========================
    # STEP 2 — CHOOSE MODEL
    # =========================
    if st.session_state.ta_step == 2:
        card2 = _ta_step_card("2) Choose model")
        with card2:
            freq = [0] * 28
            MODEL = ""
            MODELS = {}

            categories = [
                "admiration", "amusement", "anger", "annoyance", "approval", "caring",
                "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
                "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
                "nervousness", "optimism", "pride", "realization", "relief", "remorse",
                "sadness", "surprise", "neutral",
            ]

            cols = st.columns(2)
            with cols[0]:
                st.session_state.selected_option = st.radio(
                    "Select classification model type",
                    ["Recommended Models", "Search on Huggingface"],
                    key="ta_model_type_radio",
                )

            with cols[1]:
                if st.session_state.selected_option == "Recommended Models":
                    MODELS_ALL = {
                        "Covid offensive tweets Detection": {"model": "covid-twitter-bert"},
                        "Sentiment Analysis": {
                            "tokenizer": AutoTokenizer,
                            "model": "cardiffnlp/twitter-roberta-base-sentiment",
                            "id2label": {0: "Negative", 1: "Neutral", 2: "Positive"},
                        },
                        "Toxic Content Detection": {"tokenizer": AutoTokenizer, "model": "unitary/toxic-bert"},
                        "Hate Speech Detection": {"tokenizer": AutoTokenizer, "model": "cardiffnlp/twitter-roberta-base-hate-latest"},
                        "Cyberbully Detection": {"tokenizer": AutoTokenizer, "model": "sreeniketh/cyberbullying_sentiment_dsce_2023"},
                        "Named Entity Recognition": {"tokenizer": AutoTokenizer, "model": "dslim/bert-base-NER"},
                        "Parts of Speech": {"tokenizer": AutoTokenizer, "model": "QCRI/bert-base-multilingual-cased-pos-english"},
                        "Emotion Analysis": {"tokenizer": AutoTokenizer, "model": "arpanghoshal/EmoRoBERTa"},
                    }

                    selected_model_name = st.selectbox("Select a model", list(MODELS_ALL.keys()), key="ta_model_select_reco")
                    if st.session_state.current_model != selected_model_name:
                        st.session_state.predict = False
                        st.session_state.current_model = selected_model_name
                        st.session_state.output = pd.DataFrame()

                    MODELS = MODELS_ALL[selected_model_name]
                    MODEL = MODELS["model"]
                    st.session_state.disabled = False

                else:
                    st.info("Add your Huggingface API token in `secrets.toml`, then search models below.")
                    choice = st.radio("Select an option:", ["Use a specific model", "Search by keyword"], key="ta_hf_choice")

                    if choice == "Use a specific model":
                        model_name = st.text_input("Enter the huggingface model name (not URL):", key="ta_hf_model_name")
                        if st.button("Select", key="ta_hf_select_btn") and model_name:
                            st.session_state.model_list = [model_name]
                            st.session_state.disabled = False
                            if st.session_state.current_model != model_name:
                                st.session_state.predict = False
                                st.session_state.current_model = model_name
                                st.session_state.output = pd.DataFrame()

                        if st.session_state.model_list:
                            MODEL = st.radio("Model:", [st.session_state.model_list[0]], key="ta_hf_model_radio_single")
                            st.write(f"Verify model: [{MODEL}](https://huggingface.co/{MODEL})")

                    else:
                        search_text = st.text_input("Enter model keyword", key="ta_hf_search_text")
                        if st.button("Search", key="ta_hf_search_btn") and search_text:
                            st.session_state.model_list = fetch_models_from_hf(search_text)
                            st.session_state.disabled = False

                        if st.session_state.model_list:
                            MODEL = st.radio("Top Three Models:", st.session_state.model_list[:3], key="ta_hf_top3")
                            if st.checkbox("Show more", key="ta_hf_show_more"):
                                MODEL = st.radio("All Results", st.session_state.model_list[3:], key="ta_hf_all")

                            if st.session_state.current_model != MODEL:
                                st.session_state.predict = False
                                st.session_state.current_model = MODEL
                                st.session_state.output = pd.DataFrame()

                        if MODEL:
                            with st.expander("Model Details"):
                                st.write(f"Model URL: [{MODEL}](https://huggingface.co/{MODEL})")

            if st.session_state.disabled:
                st.warning("Please select a model to proceed")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            spacer, back_col, next_col = st.columns([0.52, 0.24, 0.24])
            with back_col:
                if st.button("← Back", key="ta_back_2", use_container_width=True):
                    st.session_state.ta_step = 1
                    st.rerun()
            with next_col:
                if st.button("Next →", key="ta_next_2", type="primary", use_container_width=True):
                    st.session_state._ta_model = MODEL
                    st.session_state._ta_models_meta = MODELS
                    st.session_state._ta_categories = categories
                    st.session_state._ta_freq = freq
                    st.session_state.ta_step = 3
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        return

    # =========================
    # STEP 3 — PREDICT (with "stuck" fix)
    # =========================
    if st.session_state.ta_step == 3:
        card3 = _ta_step_card("3) Predict")
        with card3:
            df = st.session_state.get("_ta_df_loaded", pd.DataFrame()).copy()
            MODEL = st.session_state.get("_ta_model", "")
            MODELS = st.session_state.get("_ta_models_meta", {})
            categories = st.session_state.get("_ta_categories", [])
            freq = st.session_state.get("_ta_freq", [0] * 28)

            if df.empty:
                st.error("No dataset found. Please go back and select data.")
                if st.button("← Back", key="ta_back_3_empty", use_container_width=True):
                    st.session_state.ta_step = 1
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                return

            df = _normalize_text_column(df)
            if "text" not in df.columns:
                st.error(f"No usable text column found. Available columns: {df.columns.tolist()}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            # ✅ NOT STUCK: show back buttons here
            if MODEL == "covid-twitter-bert":
                st.error(
                    "Covid offensive tweets Detection is disabled on this setup because it loads a .bin file "
                    "using torch.load(), which is blocked. Please select another model."
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("← Back to Choose model", key="ta_back_from_covid", use_container_width=True):
                        st.session_state.ta_step = 2
                        st.rerun()
                with c2:
                    if st.button("← Back to Select data", key="ta_back_from_covid_data", use_container_width=True):
                        st.session_state.ta_step = 1
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)
                return

            already_predicted = (
                st.session_state.get("output") is not None
                and isinstance(st.session_state.output, pd.DataFrame)
                and not st.session_state.output.empty
            )

            if not already_predicted:
                if st.button("Predict", key="ta_predict_btn", type="primary"):
                    placeholder = st.empty()
                    st.session_state.predict = True

                    with st.spinner("Downloading necessary models. It may take a few minutes..."):
                        tokenizer = AutoTokenizer.from_pretrained(MODEL)

                        if isinstance(MODELS, dict) and "id2label" in MODELS:
                            model = _safe_from_pretrained_sequence(MODEL, id2label=MODELS["id2label"])
                            tokenizer = MODELS["tokenizer"].from_pretrained(MODEL)

                        elif MODEL == "dslim/bert-base-NER":
                            model = _safe_from_pretrained_token(MODEL)
                            nlp = tpipeline("ner", model=model, tokenizer=tokenizer)

                        elif MODEL == "QCRI/bert-base-multilingual-cased-pos-english":
                            model = _safe_from_pretrained_token(MODEL)
                            pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)

                        elif MODEL == "arpanghoshal/EmoRoBERTa":
                            try:
                                model = AutoModelForSequenceClassification.from_pretrained(
                                    MODEL, from_tf=True, use_safetensors=True
                                )
                            except Exception:
                                model = AutoModelForSequenceClassification.from_pretrained(MODEL, from_tf=True)
                        else:
                            model = _safe_from_pretrained_sequence(MODEL)

                    total_rows = df.shape[0]
                    progress_bar = st.empty()

                    for index, row in df.iterrows():
                        if pd.isnull(row["text"]):
                            continue

                        if MODEL in ["dslim/bert-base-NER", "QCRI/bert-base-multilingual-cased-pos-english"]:
                            output = nlp(row["text"]) if MODEL == "dslim/bert-base-NER" else pipeline(row["text"])

                            predicted_entities = {}
                            for entity_info in output:
                                entity_group = entity_info.get("entity")
                                word = entity_info.get("word")
                                predicted_entities.setdefault(entity_group, []).append(word)

                            for punctuation in string.punctuation:
                                predicted_entities.pop(punctuation, None)

                            for entity_group, words_list in predicted_entities.items():
                                if entity_group not in df.columns:
                                    df[entity_group] = ""
                                df.at[index, entity_group] = ", ".join(words_list)

                            with placeholder.container():
                                progress = (index + 1) / total_rows
                                progress_bar.progress(progress, text=f"Predicting: {progress * 100:.2f}%")
                            continue

                        if MODEL == "arpanghoshal/EmoRoBERTa":
                            predicted_value = predict(row["text"], model, tokenizer)
                            df.loc[index, "emotion"] = predicted_value
                            if predicted_value in categories:
                                freq[categories.index(predicted_value)] += 1

                            with placeholder.container():
                                progress = (index + 1) / total_rows
                                progress_bar.progress(progress, text=f"Predicting: {progress * 100:.2f}%")
                            continue

                        predicted_value = predict(row["text"], model, tokenizer)
                        df.loc[index, "sentiment"] = predicted_value

                        with placeholder.container():
                            st.dataframe(
                                df[["text", "sentiment"]][max(0, index - 10): max(10, index)],
                                use_container_width=True,
                            )
                            progress = (index + 1) / total_rows
                            progress_bar.progress(progress, text=f"Predicting: {progress * 100:.2f}%")

                    progress_bar.empty()
                    st.session_state.output = df
                    st.success("Prediction completed", icon="✅")
                    st.rerun()
                else:
                    st.caption("Click **Predict** to run the model on your dataset.")

            if already_predicted:
                st.dataframe(st.session_state.output, use_container_width=True, height=360)

                spacer, back_col, next_col = st.columns([0.52, 0.24, 0.24])
                with back_col:
                    if st.button("← Back", key="ta_back_3", use_container_width=True):
                        st.session_state.ta_step = 2
                        st.rerun()
                with next_col:
                    if st.button("Next →", key="ta_next_3", type="primary", use_container_width=True):
                        st.session_state.ta_step = 4
                        st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        return

    # =========================
    # STEP 4 — EXPORT
    # =========================
    if st.session_state.ta_step == 4:
        card4 = _ta_step_card("4) Export results")
        with card4:
            if st.session_state.output is None or st.session_state.output.empty:
                st.warning("No predictions to export yet. Run prediction first.")
                if st.button("← Back", key="ta_back_4_empty", use_container_width=True):
                    st.session_state.ta_step = 3
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                return

            filename = st.text_input("Enter file name to save predicted data", key="ta_export_filename")
            if st.button("Save File", key="ta_save_btn") and filename:
                file_path = save_file(st.session_state.output, filename)
                st.session_state.predict = False
                st.success("Saved to '" + file_path + "'")

            csv_data = get_csv_string(st.session_state.output)
            st.download_button(
                label="Download",
                data=csv_data,
                file_name=f"{filename or 'predicted'}.csv",
                mime="text/csv",
                help="Click to download the CSV file with predicted data.",
                key="ta_download_btn",
            )

            spacer, back_col = st.columns([0.82, 0.18])
            with back_col:
                if st.button("← Back", key="ta_back_4", use_container_width=True):
                    st.session_state.ta_step = 3
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("</div>", unsafe_allow_html=True)