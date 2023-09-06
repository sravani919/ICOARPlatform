
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

# Create a subtab for "Text Annotation"
def text_annotation_tab():
    page = TextClassificationPage(st.subheader("Annotate your own data"))
    page.render()

