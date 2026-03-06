# tabs/Text_Annotation/Text_annotation.py

from typing import Dict, List
import pandas as pd
import streamlit as st

from gpt.layout import BasePage
from gpt.prompts import make_classification_prompt


class TextClassificationPage(BasePage):
    example_path = "text_classification.json"

    if "output" not in st.session_state:
        st.session_state.output = pd.DataFrame()
    if "predict" not in st.session_state:
        st.session_state.predict = False

    def make_prompt(self, examples: List[Dict]):
        return make_classification_prompt(examples)

    def prepare_inputs(self, columns: List[str]):
        return {"input": st.text_area(label="Please enter your text.", value="", height=300)}


def text_annotation_tab(labeling_mode="Text Labeling"):
    # ✅ Heading changes based on the mode
    title = "Image Labeling" if labeling_mode == "Image Labeling" else "Text Annotation"
    subtitle = (
        "Label images using your custom labels (optionally with GPT)."
        if labeling_mode == "Image Labeling"
        else "Annotate text using your custom labels (optionally with GPT)."
    )

    st.subheader(title)
    st.caption(subtitle)

    # Pass a stable parent container (better than st.subheader return object)
    parent = st.container()
    page = TextClassificationPage(parent)
    page.render(labeling_mode)