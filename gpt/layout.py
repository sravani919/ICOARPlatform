import glob
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate

from gpt.components import display_download_button, openai_model_form, task_instruction_editor, usage
from gpt.utils import escape_markdown


class BasePage(ABC):
    example_path: str = ""

    def __init__(self, title: str) -> None:
        self.title = title

    @property
    def columns(self) -> List[str]:
        return []

    def load_examples(self, filename: str) -> pd.DataFrame:
        filepath = Path(__file__).parent.resolve().joinpath("examples", filename)
        return pd.read_json(filepath)

    def make_examples(self, columns: List[str]) -> List[Dict]:
        df = self.load_examples(self.example_path)
        edited_df = st.data_editor(df, num_rows="dynamic", width=1000)
        examples = edited_df.to_dict(orient="records")
        return examples

    @abstractmethod
    def make_prompt(self, examples: List[Dict]) -> FewShotPromptTemplate:
        raise NotImplementedError()

    @abstractmethod
    def prepare_inputs(self, columns: List[str]) -> Dict:
        raise NotImplementedError()

    def annotate(self, examples: List[Dict]) -> List[Dict]:
        return examples

    def render(self) -> None:
        st.title(self.title)
        st.header("Annotate your data")
        columns = self.columns
        examples = self.make_examples(columns)
        examples = self.annotate(examples)

        prompt = self.make_prompt(examples)
        prompt = task_instruction_editor(prompt)

        st.header("Test")
        col1, col2 = st.columns([3, 1])

        with col1:
            inputs = self.prepare_inputs(columns)

        with col2:
            llm = openai_model_form()

        with st.expander("See your prompt"):
            st.markdown(f"```\n{prompt.format(**inputs)}\n```")

        if llm is None:
            st.error("Enter your API key.")

        if st.button("Predict", disabled=llm is None):
            chain = LLMChain(llm=llm, prompt=prompt)  # type:ignore
            response = chain.run(**inputs)
            st.markdown(escape_markdown(response).replace("\n", "  \n"))

            chain.save("config.yaml")
            display_download_button()

        usage()

        st.header("Label Your Data")
        option = st.selectbox("Select a file", [file for file in glob.glob("./data/*.csv")])
        if st.button("Predict Labels"):
            st.session_state.predict = True
            st.session_state.filename_pred = option
            df = pd.read_csv(option)
            total_rows = df.shape[0]

            progress_bar = st.empty()

            for index, row in df.iterrows():
                progress = (index + 1) / total_rows
                progress_bar.progress(progress, text=f"Predicting text: {progress * 100:.2f}% complete")

            st.dataframe(df)
            progress_bar.empty()

            st.session_state.output = df
            st.success("Prediction completed", icon="✅")

        if st.session_state.predict:
            filename = st.text_input("Enter file name  to save predicted data")
            save = st.button("Save File")
            if save:
                file_path = save_file(st.session_state.output, filename)
                st.session_state.predict = False
                st.success("Saved to '" + file_path + "'")


def save_file(df, filename):
    if not os.path.exists("predicted"):
        os.makedirs("predicted")
    file_path = f"predicted/{filename}.csv"
    df.to_csv(file_path, index=False)
    return file_path
