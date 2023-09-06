import glob
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate

from gpt.components import display_download_button, openai_model_form, task_instruction_editor
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
    
        # st.title(self.title)
        # st.divider()
        # st.header("Annotate your own data")
        st.markdown("""In this section, you have the opportunity to utilize ChatGPT to annotate text according to your 
                    custom labels. You have the option to label a CSV file using the provided ChatGPT interface. Below 
                    is a demo of how it works:
                    """)
        st.markdown("In this example, we are using sentiment labels on a text:")
        st.markdown("""**1. Input Text:** Enter or paste the text you want to annotate into the provided text box along with 
                    desired labels. This text may include sentences, paragraphs, epending on your annotation needs.""")

        columns = self.columns
        examples = self.make_examples(columns)
        examples = self.annotate(examples)

        st.markdown("""**2. Edit the prompt (optional):** You can modify the prompt that instruct ChatGPT. The choice of 
                    prompt greatly influences the generated responsesand the quality of the annotation. Ensure that your
                    edits are clear and relevant to the task.""")
        
        prompt = self.make_prompt(examples)
        prompt = task_instruction_editor(prompt)

        st.header("Test")
        # col1, col2 = st.columns([3, 1])

        inputs = self.prepare_inputs(columns)

        with st.sidebar:
            llm = openai_model_form()
        # with col1:
        #     inputs = self.prepare_inputs(columns)
        #
        # with col2:
        #     llm = openai_model_form()

        with st.expander("See your full prompt"):
            st.markdown(f"```\n{prompt.format(**inputs)}\n```")

        if llm is None:
            st.error("Enter your API key.")

        if st.button("Predict", disabled=llm is None):
            chain = LLMChain(llm=llm, prompt=prompt)  # type:ignore
            response = chain.run(**inputs)
            st.markdown(escape_markdown(response).replace("\n", "  \n"))
            chain.save("config.yaml")
            display_download_button()

        # usage()

        # section for chatgpt labeling a dataset created by icoar
        st.header("Label Your Data")

        st.text("This will use the parameters you input in the test section.")

        option = st.selectbox("Select a file", [file for file in glob.glob("./data/*.csv")],key="unique_key_1")

        if llm is None:
            st.error("Enter your API key.")

        if st.button("Predict Labels", disabled=llm is None):
            st.session_state.predict = True
            st.session_state.filename_pred = option
            df = pd.read_csv(option)

            total_rows = df.shape[0]
            progress_bar = st.empty()

            chain = LLMChain(llm=llm, prompt=prompt)

            for index, row in df.iterrows():
                retries = 5
                while retries > 0:
                    try:
                        response = chain.run(row["text"])
                        response = response.replace("label:", "").strip().capitalize()
                        df.loc[index, "label"] = response
                        progress = (index + 1) / total_rows
                        progress_bar.progress(progress, text=f"Predicting text: {progress * 100:.2f}% complete")
                        break
                    except Exception as ex:
                        print(ex)
                        retries -= 1
                        time.sleep(10)

            st.dataframe(df)
            progress_bar.empty()

            st.session_state.output = df
            st.success("Prediction completed", icon="✅")

        if st.session_state.predict:
            filename = st.text_input("Enter file name to save predicted data")
            save = st.button("Save File")
            if save:
                if not os.path.exists("predicted"):
                    os.makedirs("predicted")
                file_path = f"predicted/{filename}.csv"
                st.session_state.output.to_csv(file_path, index=False)

                st.session_state.predict = False
                st.success("Saved to '" + file_path + "'")


        
