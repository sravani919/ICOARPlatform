import base64
import glob
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from PIL import Image
from streamlit import secrets

from gpt.components import display_download_button, openai_model_form, task_instruction_editor
from gpt.utils import escape_markdown


class BasePage(ABC):
    example_path: str = ""

    def __init__(self, title: str) -> None:
        self.title = title
        if "openai" in secrets:
            self.api_key = next(iter(secrets["openai"].values()), None)

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

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def prepare_images(self, images_list, folder_path):
        image_messages = []
        for _, row in images_list.iterrows():
            image_path = os.path.join(folder_path, row["Image Name"])
            label = row["Label"]

            base64_image = self.encode_image(image_path)

            image_message = {
                "type": "text",
                "text": f"Label for image: {label}",
            }
            image_url_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }

            image_messages.append(image_message)
            image_messages.append(image_url_message)

        return image_messages

    def text_labeling(self):
        if "llm" not in st.session_state:
            st.session_state.llm = None

        if "predict" not in st.session_state:
            st.session_state.predict = False
        st.markdown(
            f"""
                    <style>
                        div[data-testid="stButton"] {{
                            margin-top: {8}px;
                            margin-bottom: {6}px;
                        }}
                        div[data-testid="stMarkdownContainer"] {{
                            margin-top: {2}px;
                            margin-bottom: {2}px;
                        }}
                    </style>
                    """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """In this section, you have the opportunity to utilize ChatGPT to annotate text according to your
                    custom labels. You have the option to label a CSV file using the provided ChatGPT interface. Below
                    is a demo of how it works:
                    """
        )
        st.markdown("In this example, we are labeling whether text is cyberbullying or not:")

        st.markdown(
            """**1: Input Your API Key**
                Use your API key and store it in the secrets.toml file like so:\n\n[openai]\n\napi_key = (your key)"""
        )

        st.markdown(
            """**2. Input Text:** Enter or paste the text you want to annotate into the provided text box along with
                    desired labels. This text may include sentences, paragraphs, or any content based on your custom
                    labels.Along with the text, specify the labels you want to apply."""
        )

        columns = self.columns
        self.example_path = "text_classification.json"
        examples = self.make_examples(columns)
        examples = self.annotate(examples)

        st.markdown(
            """**3. Edit the prompt (Optional):** You can modify the prompt that instruct ChatGPT. The choice of
                    prompt greatly influences the generated responsesand the quality of the annotation. Ensure that your
                    edits are clear and relevant to the task."""
        )

        prompt = self.make_prompt(examples)
        prompt = task_instruction_editor(prompt)

        st.markdown(
            """**4. Test with a Single Example (Optional):** Before labeling your entire dataset, it's a good practice
            to test ChatGPT performance with a single example to see if the predictions are accurate and align with your
            instructions."""
        )

        inputs = self.prepare_inputs(columns)

        with st.expander("See your full prompt"):
            st.markdown(f"```\n{prompt.format(**inputs)}\n```")

        st.markdown(
            """**5. Parameter Usage:** The parameters you specify means you can make adjustments to how ChatGPT
            generates responses to better suit your task. For meanings of sliders see
            [this link](https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api-a-few-tips-and-tricks-on-controlling-the-creativity-deterministic-output-of-prompt-responses/172683)."""
        )

        if self.api_key is None:
            st.error("Enter your API key in secrets.toml under [openai].")
            return

        with st.expander("Add hyper-parameters"):
            st.session_state.llm = openai_model_form()

        if st.button("Predict", disabled=st.session_state.llm is None):
            chain = LLMChain(llm=st.session_state.llm, prompt=prompt)  # type:ignore
            response = chain.run(**inputs)
            st.markdown(escape_markdown(response).replace("\n", "  \n"))
            chain.save("config.yaml")
            display_download_button()

        #  usage()

        # section for chatgpt labeling a dataset created by icoar
        st.markdown(
            """**5. Upload the File:** Once you are satisfied with the test results, you can proceed to label your
            entire dataset. Click the "Select a File" button to upload the file you want to label. Please ensure that
            your CSV file is properly formatted the labels you provided in Step 1 should match the labels in your CSV
            file for accurate annotation."""
        )

        st.markdown("**Note:** This will use the parameters you put in the test section.")

        if st.session_state.llm is None:
            st.error("Enter your API key.")

        option = st.selectbox("Select a file", [file for file in glob.glob("./data/*.csv")], key="unique_key_1")

        if st.button("Predict Labels", disabled=st.session_state.llm is None):
            st.session_state.predict = True
            st.session_state.filename_pred = option
            df = pd.read_csv(option)

            total_rows = df.shape[0]
            progress_bar = st.empty()

            chain = LLMChain(llm=st.session_state.llm, prompt=prompt)

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
            username = st.session_state["username"]
            if save:
                if not os.path.exists("predicted"):
                    os.makedirs("predicted")
                    os.makedirs(f"""predicted/{username}""")
                file_path = f"predicted/{username}/{filename}.csv"
                st.session_state.output.to_csv(file_path, index=False)

                st.session_state.predict = False
                st.success("Saved to '" + file_path + "'")

    def image_labeling(self):
        if "predict" not in st.session_state:
            st.session_state.predict = False
        st.markdown(
            f"""
                    <style>
                        div[data-testid="stButton"] {{
                            margin-top: {8}px;
                            margin-bottom: {6}px;
                        }}
                        div[data-testid="stMarkdownContainer"] {{
                            margin-top: {2}px;
                            margin-bottom: {2}px;
                        }}
                    </style>
                    """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """In this section, you have the opportunity to utilize ChatGPT to annotate images according to your
            custom labels. In the preset example below, we are labeling an image based on  whether it is
            cyberbullying or not."""
        )

        st.markdown(
            """**1: Input Your API Key**
        Use your API key and store it in the secrets.toml file like so:\n\n[openai]\n\napi_key = (your key)"""
        )

        st.markdown(
            """**2. Input Image(s):** Choose the folder that contains the images you want to use as a sample,
            and label those images. The program is looking specifically in the data folder of the project.
            The demo has some preset images that are already labeled."""
        )
        # insert example images, also give users options to choose a folder here

        data_directory = "./data/"

        current_file_directory = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the demo directory
        default_folder = os.path.join(current_file_directory, "examples/demo")
        relative_demo_path = os.path.relpath(default_folder, current_file_directory)
        # Initialize an empty list to store subdirectories with full paths
        subdirectories = [relative_demo_path]

        # Iterate over the list of directories
        for d in os.listdir(data_directory):
            full_path = os.path.join(data_directory, d)

            # Check if the item is a directory
            if os.path.isdir(full_path):
                # Append the full path to the subdirectories list
                subdirectories.append(full_path)

        # Allow the user to choose a folder with demo
        selected_folder_path = st.selectbox("Select a folder that contains images", subdirectories, key="unique_key_1")

        # Get the full path for the selected folder
        option = selected_folder_path
        if option == relative_demo_path:
            option = default_folder

        images_list = [
            image for image in os.listdir(option) if image.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        ]

        # Create a DataFrame to store labels for each image
        image_labels_df = pd.DataFrame(columns=["Image Name", "Label"])
        image_labels_df["Image Name"] = images_list

        if option == default_folder:
            image_labels_df.loc[image_labels_df["Image Name"].isin(["CB 1.png", "CB 2.png"]), "Label"] = "1"
            image_labels_df.loc[image_labels_df["Image Name"].isin(["NCB 1.png", "NCB 2.png"]), "Label"] = "0"

        # Use st.data_editor to allow users to label images in a table-like interface
        edited_labels = st.data_editor(key="labeling_editor", data=image_labels_df, num_rows="dynamic", width=1000)

        st.markdown(
            """**3. Edit the prompt (Optional):** You can modify the prompt that instructs ChatGPT. The choice of
            prompt greatly influences the generated responses and the quality of the annotation. Ensure that your
            edits are clear and relevant to the task."""
        )

        # Add a text area for prompt editing
        st.subheader("Edit Prompt:")
        prompt = st.text_area(
            "Customize your prompt:",
            value="Q: Label this image as cyberbullying based on the sample"
            " provided, answering 0 for 'No' and 1 for 'Yes' Give "
            "only the number.",
            height=100,
        )

        if "openai" not in secrets:
            st.error("Enter your API key in secrets.toml under [openai].")
            return

        # do a section where you predict one image
        st.markdown(
            """**4. Test with a Single Example (Optional):** Before labeling your entire dataset, it's a good practice
            to test ChatGPT performance with a single example to see if the predictions are accurate and align with your
            instructions."""
        )
        # allow the user to choose an image to predict, first selecting a folder and then an image
        selected_folder_path2 = st.selectbox("Select a folder that contains images", subdirectories, key="unique_key_2")

        # Get the full path for the selected folder
        option2 = selected_folder_path2
        if option2 == relative_demo_path:
            current_file_directory = os.path.dirname(os.path.abspath(__file__))
            option2 = os.path.join(current_file_directory, "examples/demo_test")

        image_files = [f for f in os.listdir(option2) if os.path.isfile(os.path.join(option2, f))]

        image_test_name = st.selectbox("Select an image to predict", image_files)

        image_test_path = os.path.join(option2, image_test_name)

        image = Image.open(image_test_path)
        image = image.resize((200, 200))
        st.image(image, caption="Image to predict")

        base64_image = self.encode_image(image_test_path)

        if st.button("Predict Label", disabled="openai" not in secrets):
            prepared_images = self.prepare_images(edited_labels, option)
            # Call GPT Vision API for image labeling
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            *prepared_images,
                            {
                                "type": "text",
                                "text": "Use these images and their labels to answer the following prompt:",
                            },
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "Here is the image to label based on the prompt:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                "max_tokens": 50,
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            data = response.json()
            content_value = data["choices"][0]["message"]["content"]
            st.markdown(f"**Predicted Label:** {content_value}")

        st.markdown(
            """**4. Test with a Multiple Images:** Select a folder of images to label and then
            click the button to predict the labels for all the images in the folder."""
        )

        label_subdirectories = [x for x in subdirectories if x != relative_demo_path]
        label_folder_path = st.selectbox(
            "Select a folder that contains images", label_subdirectories, key="unique_" "key_3"
        )

        # Get the full path for the selected folder
        option_label = label_folder_path

        if st.button("Predict Labels", disabled="openai" not in secrets):
            st.session_state.predict = True
            st.session_state.filename_pred = option

            prepared_images = self.prepare_images(edited_labels, option)
            image_files = [f for f in os.listdir(option_label) if os.path.isfile(os.path.join(option_label, f))]
            # Create a DataFrame with the names of the files
            df = pd.DataFrame({"Image Name": image_files})

            total_rows = df.shape[0]
            progress_bar = st.empty()

            for index, row in df.iterrows():
                image_name = row["Image Name"]  # Replace with the column containing image paths
                image_path = os.path.join(option_label, image_name)

                if not os.path.isfile(image_path) or not image_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".gif", ".bmp")
                ):
                    continue

                base64_image = self.encode_image(image_path)

                # Call GPT Vision API for image labeling
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
                payload = {
                    "model": "gpt-4-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                *prepared_images,
                                {
                                    "type": "text",
                                    "text": "Use these images and their labels to answer the following prompt:",
                                },
                                {"type": "text", "text": prompt},
                                {"type": "text", "text": "Here is the image to label based on the prompt:"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            ],
                        }
                    ],
                    "max_tokens": 50,
                }
                try:
                    time.sleep(10)
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
                    )
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                    data = response.json()
                    content_value = data["choices"][0]["message"]["content"]

                    # Update DataFrame with predicted label
                    df.loc[index, "predicted_label"] = content_value
                    progress = (index + 1) / total_rows
                    progress_bar.progress(progress, text=f"Predicting image labels: {progress * 100:.2f}% complete")
                except requests.exceptions.RequestException as e:
                    st.error(
                        f"An error occurred while predicting the label for image {image_name}: {e}. Please retry "
                        f"the prediction."
                    )

            st.dataframe(df)
            progress_bar.empty()

            st.session_state.output = df
            st.success("Prediction completed", icon="✅")

        if st.session_state.predict:
            filename = st.text_input("Enter file name to save predicted data")
            save = st.button("Save File")
            username = st.session_state["username"]
            if save:
                if not os.path.exists("predicted"):
                    os.makedirs("predicted")
                os.makedirs(f"""predicted/{username}""")
                file_path = f"predicted/{username}/{filename}.csv"
                st.session_state.output.to_csv(file_path, index=False)

                st.session_state.predict = False
                st.success("Saved to '" + file_path + "'")

    def render(self) -> None:
        labeling_mode = st.selectbox("Select Labeling Mode", ["Text Labeling", "Image Labeling"])

        if labeling_mode == "Text Labeling":
            self.text_labeling()
        elif labeling_mode == "Image Labeling":
            self.image_labeling()
