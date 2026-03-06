import base64
import os
import zipfile
from pathlib import Path
from typing import List

import requests
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras

DATA_DIR = Path("data/images/image")  # consistent path


# -----------------------------
# TF image utils
# -----------------------------
def preprocess_image(image_bytes):
    # ✅ works for jpg/png (instead of decode_png only)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    return image


def load_and_preprocess_image(path):
    image_bytes = tf.io.read_file(path)
    return preprocess_image(image_bytes)


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


# -----------------------------
# Cache helpers
# -----------------------------
def empty_cache():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for p in DATA_DIR.glob("*"):
        if p.is_file():
            p.unlink()
    st.session_state.image_uploaded = False


@st.cache_resource(show_spinner=False)
def _load_vgg_model():
    return keras.models.load_model("model/fine_tuned_vgg16_model.h5")


def _mm_subcard(title: str):
    wrap = st.container()
    with wrap:
        st.markdown('<div class="mm-subcard-marker"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mm-subtitle">{title}</div>', unsafe_allow_html=True)
    return wrap


def _saved_image_paths() -> List[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([str(p) for p in DATA_DIR.glob("*.jpg")] + [str(p) for p in DATA_DIR.glob("*.png")])


def _save_upload_to_data_dir(file_):
    """
    Save uploaded jpg/png or zip into DATA_DIR.
    Returns list of saved image paths.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # clear previous
    if any(DATA_DIR.glob("*")):
        empty_cache()

    if file_.type == "application/zip":
        save_path = DATA_DIR / "images.zip"
        save_path.write_bytes(file_.getbuffer())
        with zipfile.ZipFile(save_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        save_path.unlink(missing_ok=True)
        st.session_state.image_uploaded = True
        st.success("Zip extracted successfully!")
    else:
        out = DATA_DIR / file_.name
        out.write_bytes(file_.read())
        st.session_state.image_uploaded = True
        st.success("Image saved successfully!")

    return _saved_image_paths()


# =========================================================
# 1) VGG Cyberbullying vs Non-cyberbullying (Wizard)
# =========================================================
def bully_classification():
    if "bully_step" not in st.session_state:
        st.session_state.bully_step = 1
    if "image_uploaded" not in st.session_state:
        st.session_state.image_uploaded = False

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # STEP 1 — UPLOAD
    # =========================
    if st.session_state.bully_step == 1:
        card = _mm_subcard("1) Upload image(s)")
        with card:
            file_ = st.file_uploader(
                "Upload an image (.jpg/.png) or a .zip of images.",
                type=["jpg", "png", "zip"],
                key="bully_upload",
            )

            cols = st.columns([0.70, 0.15, 0.15])
            with cols[1]:
                if st.button("Reset", use_container_width=True, key="bully_reset"):
                    empty_cache()
                    st.session_state.bully_step = 1
                    st.rerun()

            if file_ is not None:
                _save_upload_to_data_dir(file_)

            spacer, next_col = st.columns([0.76, 0.24])
            with next_col:
                if st.button(
                    "Next →",
                    type="primary",
                    use_container_width=True,
                    disabled=not st.session_state.image_uploaded,
                    key="bully_next",
                ):
                    st.session_state.bully_step = 2
                    st.rerun()
        return

    # =========================
    # STEP 2 — PREDICT
    # =========================
    if st.session_state.bully_step == 2:
        card = _mm_subcard("2) Predict")
        with card:
            image_paths = _saved_image_paths()

            if not image_paths:
                st.warning("No images found. Go back and upload again.")
                if st.button("← Back", use_container_width=False, key="bully_back_to_upload"):
                    st.session_state.bully_step = 1
                    st.rerun()
                return

            st.info(f"Found {len(image_paths)} image(s). Ready to run prediction.")

            spacer, back_col, run_col = st.columns([0.58, 0.18, 0.24])
            with back_col:
                if st.button("← Back", use_container_width=True, key="bully_back"):
                    st.session_state.bully_step = 1
                    st.rerun()
            with run_col:
                if st.button("Run Prediction", type="primary", use_container_width=True, key="bully_run"):
                    st.session_state["bully_image_paths"] = image_paths
                    st.session_state.bully_step = 3
                    st.rerun()
        return

    # =========================
    # STEP 3 — RESULTS
    # =========================
    if st.session_state.bully_step == 3:
        card = _mm_subcard("3) Results")
        with card:
            image_paths = st.session_state.get("bully_image_paths", [])
            if not image_paths:
                st.warning("No cached images found. Go back.")
                if st.button("← Back", use_container_width=False, key="bully_back_no_cache"):
                    st.session_state.bully_step = 1
                    st.rerun()
                return

            model = _load_vgg_model()
            class_names = ["cyberbullying", "non_cyberbullying"]

            # dataset
            label_ids = [0] * len(image_paths)
            ds = tf.data.Dataset.from_tensor_slices((image_paths, label_ids))
            ds = ds.map(load_and_preprocess_from_path_label).batch(16)

            # ✅ predict ONCE for all images
            all_preds = []
            for batch_imgs, _ in ds:
                preds = model.predict(batch_imgs, verbose=0)
                all_preds.extend(tf.argmax(preds, axis=1).numpy().tolist())

            st.subheader("Predictions")
            st.divider()

            # grid
            cols = st.columns(3)
            for idx, path in enumerate(image_paths):
                c = cols[idx % 3]
                with c:
                    pred_idx = int(all_preds[idx])
                    label = class_names[pred_idx]
                    color_style = "red" if label == "cyberbullying" else "green"
                    st.markdown(
                        f"<div style='font-weight:900;color:{color_style};margin-bottom:6px'>{label}</div>",
                        unsafe_allow_html=True,
                    )
                    img = Image.open(path).convert("RGB")
                    st.image(img, use_container_width=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            spacer, back_col = st.columns([0.82, 0.18])
            with back_col:
                if st.button("← Back", use_container_width=True, key="bully_back_from_results"):
                    st.session_state.bully_step = 2
                    st.rerun()


# =========================================================
# 2) Cyberbullying Detection using GPT (LLM)
# =========================================================
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_classification_llm():
    card = _mm_subcard("1) Upload image(s)")

    with card:
        if "image_uploaded" not in st.session_state:
            st.session_state.image_uploaded = False

        file_ = st.file_uploader(
            "Upload an image (.jpg/.png) or a .zip of images.",
            type=["zip", "jpg", "png"],
            key="gpt_upload",
        )

        cols = st.columns([0.70, 0.15, 0.15])
        with cols[1]:
            if st.button("Reset", use_container_width=True, key="gpt_reset"):
                empty_cache()
                st.session_state.image_uploaded = False
                st.rerun()

        if file_ is not None:
            _save_upload_to_data_dir(file_)

    image_paths = _saved_image_paths()
    if not image_paths:
        st.info("Upload an image to begin.")
        return

    # ✅ your secrets style (same as your old code)
    try:
        api_key = st.secrets.openai.openAI
    except Exception:
        st.error("OpenAI API key not found. Add it in secrets.toml under [openai] openAI='YOUR_KEY'.")
        return

    card2 = _mm_subcard("2) Run GPT Classification")
    with card2:
        if st.button("Run", type="primary", use_container_width=False, key="gpt_run"):
            st.session_state["gpt_results"] = []

            for image_path in image_paths:
                base64_image = encode_image(image_path)

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }

                prompt = (
                    "Label the following image with one of: Cyberbullying or Non-Cyberbullying.\n"
                    "Cyberbullying: if the image can be used to bully, threaten, harass, or scare someone.\n"
                    "Non-Cyberbullying: otherwise.\n"
                    "Return only one word: Cyberbullying or Non-Cyberbullying."
                )

                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            ],
                        }
                    ],
                    "max_tokens": 20,
                }

                try:
                    r = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=60,
                    )
                    r.raise_for_status()
                    data = r.json()
                    label = data["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    label = "ERROR"
                    st.error(f"GPT call failed for {os.path.basename(image_path)}")
                    st.exception(e)

                st.session_state["gpt_results"].append((image_path, label))

    results = st.session_state.get("gpt_results", None)
    if not results:
        return

    card3 = _mm_subcard("3) Results")
    with card3:
        cols = st.columns(3)
        for i, (image_path, label) in enumerate(results):
            c = cols[i % 3]
            with c:
                color_style = "red" if "Cyberbullying" in label else "green"
                st.markdown(
                    f"<div style='font-weight:900;color:{color_style};margin-bottom:6px'>{label}</div>",
                    unsafe_allow_html=True,
                )
                st.image(Image.open(image_path).convert("RGB"), use_container_width=True)